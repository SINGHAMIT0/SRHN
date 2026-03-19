/*
 * srhn_reasoning.c  —  Multi-hop reasoning, causal chains, contradiction detection
 *
 * This module transforms raw graph activations into structured reasoning artifacts:
 *   1. Propagation  — BFS activation spread with path tracking
 *   2. Chain building — extract top-K reasoning chains from activated nodes
 *   3. Causal path  — A* search for strongest causal route between two nodes
 *   4. Contradictions — detect EDGE_CONTRADICTS among highly activated node pairs
 *   5. Hyperedge eval — fire hyperedges based on activation state
 */

#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── BFS queue entry ─────────────────────────────────────────── */
typedef struct {
    uint32_t node_id;
    float    spike;
    uint32_t parent;       /* for chain reconstruction */
    uint32_t parent_edge;  /* edge type to parent */
    float    path_strength;/* cumulative path strength */
    uint8_t  depth;
} PropQItem;

/* ── Working memory activation bonus ────────────────────────── */
static float working_mem_bonus(SRHNNetwork4 *net, uint32_t nid) {
    for (uint8_t s = 0; s < net->working_mem.n_slots; s++)
        if (net->working_mem.slot_nodes[s] == nid)
            return 10.f * net->working_mem.slot_salience[s];
    return 1.f;
}

/* ── Propagation ─────────────────────────────────────────────── */
void srhn4_propagate(SRHNNetwork4 *net, uint32_t seed, float init_spike,
                      SRHNResult4 *result) {
    if (seed >= net->n_nodes || init_spike < RESONANCE_THRESH) return;

    /* Allocate per-call state on heap — thread safe, no stack overflow */
    bool       *visited    = (bool       *)calloc(net->n_nodes, sizeof(bool));
    uint32_t   *parents    = (uint32_t   *)malloc(net->n_nodes * sizeof(uint32_t));
    float      *path_str   = (float      *)calloc(net->n_nodes, sizeof(float));
    PropQItem  *queue      = (PropQItem  *)malloc(4096 * sizeof(PropQItem));
    int         q_cap      = 4096;

    if (!visited || !parents || !path_str || !queue) {
        free(visited); free(parents); free(path_str); free(queue);
        return;
    }
    /* Initialize parents to UINT32_MAX */
    for (uint32_t i = 0; i < net->n_nodes; i++) parents[i] = UINT32_MAX;

    int head = 0, tail = 0;
    queue[tail++] = (PropQItem){seed, init_spike, UINT32_MAX, 0, init_spike, 0};
    visited[seed] = true;
    path_str[seed] = init_spike;

    while (head < tail && result->n_activated < MAX_ACTIVATED - 1) {
        PropQItem cur = queue[head++];

        float cur_spike = cur.spike;
        if (cur_spike < 0.03f) continue;

        SRHNNode4 *node = &net->nodes[cur.node_id];
        if (node->type == NODE_PRUNED) continue;

        /* Apply working memory bonus */
        float bonus = working_mem_bonus(net, cur.node_id);
        float act_delta = cur_spike * bonus;

        node->activation = srhn4_clampf(node->activation + act_delta, 0.f, 1.f);
        if (node->activation > node->activation_peak)
            node->activation_peak = node->activation;
        node->is_active   = true;
        node->usage_count += 1.f;
        node->last_active  = net->global_time;

        /* PageRank importance bonus to activation */
        if (net->pagerank.scores && cur.node_id < net->pagerank.n_scores)
            node->activation = srhn4_clampf(
                node->activation + net->pagerank.scores[cur.node_id] * 0.1f, 0.f, 1.f);

        parents[cur.node_id]  = cur.parent;
        path_str[cur.node_id] = cur.path_strength;

        result->activated_nodes[result->n_activated] = cur.node_id;
        result->activations    [result->n_activated] = node->activation;
        result->n_activated++;

        if (cur.depth >= MAX_HOPS) continue;

        float next_spike = cur_spike * SPIKE_DECAY;

        for (uint8_t i = 0; i < node->n_neighbors; i++) {
            uint32_t nb_id = node->neighbors[i];
            if (nb_id >= net->n_nodes || visited[nb_id]) continue;

            SRHNNode4 *nb = &net->nodes[nb_id];
            if (nb->type == NODE_PRUNED) continue;

            float res = srhn4_resonance(&node->sig, &nb->sig);
            float ew  = node->edge_weights[i];

            /* Inhibitory edge: reduce spike instead of boosting */
            float edge_mult = 1.f;
            if (node->edge_types && node->edge_types[i] == EDGE_INHIBITORY)
                edge_mult = -0.3f;

            float gspike = next_spike * (0.45f * res + 0.45f * ew + 0.10f) * edge_mult;

            if (fabsf(gspike) >= RESONANCE_THRESH * 0.2f) {
                visited[nb_id] = true;
                if (tail >= q_cap - 1) {
                    /* Grow queue */
                    int nc = q_cap * 2;
                    PropQItem *np = (PropQItem *)realloc(queue, nc * sizeof(PropQItem));
                    if (!np) goto propagate_done;
                    queue = np; q_cap = nc;
                }
                queue[tail++] = (PropQItem){
                    nb_id,
                    fabsf(gspike),
                    cur.node_id,
                    node->edge_types ? node->edge_types[i] : EDGE_ASSOC,
                    cur.path_strength * (0.45f * res + 0.45f * ew),
                    (uint8_t)(cur.depth + 1)
                };
            }
        }
    }

propagate_done:
    free(visited);
    free(parents);
    free(path_str);
    free(queue);
}

/* ── Chain building ──────────────────────────────────────────── */
/*
 * After propagation, reconstruct reasoning chains by following the
 * strongest paths through the activated node set.
 */
void srhn4_build_chains(SRHNNetwork4 *net, SRHNResult4 *result) {
    if (result->n_activated < 2) return;

    /* Find top N activated nodes as chain endpoints */
    typedef struct { uint32_t id; float act; } ActNode;
    ActNode top[MAX_CHAINS * 2];
    int ntop = 0;

    for (uint32_t i = 0; i < result->n_activated; i++) {
        uint32_t nid = result->activated_nodes[i];
        float act    = result->activations[i];
        if (act < 0.2f) continue;

        /* Insertion sort */
        int j = ntop;
        int max_top = MAX_CHAINS * 2;
        while (j > 0 && top[j-1].act < act && j < max_top) {
            if (j < max_top) top[j] = top[j-1];
            j--;
        }
        if (j < max_top) {
            top[j] = (ActNode){nid, act};
            if (ntop < max_top) ntop++;
        }
    }

    result->n_chains = 0;

    /* For each pair of top nodes, find path via common activated neighbors */
    for (int a = 0; a < ntop && result->n_chains < MAX_CHAINS; a++) {
        for (int b = a + 1; b < ntop && result->n_chains < MAX_CHAINS; b++) {
            uint32_t na = top[a].id;
            uint32_t nb_id = top[b].id;

            if (na >= net->n_nodes || nb_id >= net->n_nodes) continue;

            SRHNNode4 *node_a = &net->nodes[na];
            SRHNNode4 *node_b = &net->nodes[nb_id];

            /* Check if b is a neighbor of a */
            bool direct = false;
            float edge_w = 0.f;
            EdgeType etype = EDGE_ASSOC;
            for (uint8_t k = 0; k < node_a->n_neighbors; k++) {
                if (node_a->neighbors[k] == nb_id) {
                    direct    = true;
                    edge_w    = node_a->edge_weights[k];
                    if (node_a->edge_types) etype = node_a->edge_types[k];
                    break;
                }
            }

            if (!direct && top[a].act < 0.35f) continue;

            float res = srhn4_resonance(&node_a->sig, &node_b->sig);
            float strength = direct ? (0.5f * res + 0.5f * edge_w) : res * 0.6f;

            if (strength < 0.15f) continue;

            ReasoningChain *chain = &result->chains[result->n_chains++];
            memset(chain, 0, sizeof(ReasoningChain));

            chain->nodes[0] = na;
            strncpy(chain->labels[0], node_a->label, 127);
            chain->strengths[0] = top[a].act;

            if (direct) {
                chain->nodes[1] = nb_id;
                strncpy(chain->labels[1], node_b->label, 127);
                chain->strengths[1] = edge_w;
                chain->edge_types[1] = etype;
                /* Relation label from edge type */
                static const char *etype_names[] = {
                    "associated with", "causes", "synonym of", "antonym of",
                    "instance of", "part of", "leads to", "inhibits",
                    "before", "located at", "code relation", "visual link",
                    "contradicts", "supports", "unknown"
                };
                strncpy(chain->relations[0],
                        etype_names[(int)etype < 15 ? (int)etype : 14], 63);
                chain->length = 2;
            } else {
                /* Try to find a 2-hop path through activated neighbors */
                uint32_t bridge = UINT32_MAX;
                float best_bridge = 0.f;
                for (uint8_t k = 0; k < node_a->n_neighbors; k++) {
                    uint32_t mid = node_a->neighbors[k];
                    if (mid >= net->n_nodes) continue;
                    if (!net->nodes[mid].is_active) continue;
                    for (uint8_t m = 0; m < net->nodes[mid].n_neighbors; m++) {
                        if (net->nodes[mid].neighbors[m] == nb_id) {
                            float br = node_a->edge_weights[k] *
                                       net->nodes[mid].edge_weights[m];
                            if (br > best_bridge) { best_bridge = br; bridge = mid; }
                        }
                    }
                }
                if (bridge != UINT32_MAX) {
                    chain->nodes[1] = bridge;
                    strncpy(chain->labels[1], net->nodes[bridge].label, 127);
                    strncpy(chain->relations[0], "associated with", 63);
                    chain->strengths[1] = best_bridge;
                    chain->nodes[2] = nb_id;
                    strncpy(chain->labels[2], node_b->label, 127);
                    strncpy(chain->relations[1], "leads to", 63);
                    chain->strengths[2] = top[b].act;
                    chain->length = 3;
                } else {
                    chain->nodes[1] = nb_id;
                    strncpy(chain->labels[1], node_b->label, 127);
                    strncpy(chain->relations[0], "resonates with", 63);
                    chain->strengths[1] = res;
                    chain->length = 2;
                }
            }

            chain->total_strength = strength;
            chain->is_causal = (etype == EDGE_CAUSAL || etype == EDGE_CAUSES);
        }
    }
}

/* ── Causal path (A* / Dijkstra with edge type preference) ──── */
void srhn4_causal_path(SRHNNetwork4 *net, uint32_t from, uint32_t to,
                        ReasoningChain *out) {
    if (!out || from >= net->n_nodes || to >= net->n_nodes) return;
    memset(out, 0, sizeof(ReasoningChain));

    if (from == to) {
        out->nodes[0] = from;
        strncpy(out->labels[0], net->nodes[from].label, 127);
        out->length = 1; out->total_strength = 1.f;
        return;
    }

    /* Dijkstra: weight = 1 - causal edge strength (prefer causal edges) */
    float *dist   = (float    *)malloc(net->n_nodes * sizeof(float));
    uint32_t *prev = (uint32_t *)malloc(net->n_nodes * sizeof(uint32_t));
    bool  *seen   = (bool     *)calloc(net->n_nodes, sizeof(bool));
    if (!dist || !prev || !seen) {
        free(dist); free(prev); free(seen); return;
    }

    for (uint32_t i = 0; i < net->n_nodes; i++) {
        dist[i] = FLT_MAX;
        prev[i] = UINT32_MAX;
    }
    dist[from] = 0.f;

    /* Simple priority queue using linear scan (adequate for <10K nodes) */
    uint32_t iters = 0;
    while (iters++ < net->n_nodes) {
        /* Find unvisited node with minimum distance */
        float min_d = FLT_MAX; uint32_t u = UINT32_MAX;
        for (uint32_t i = 0; i < net->n_nodes; i++) {
            if (!seen[i] && dist[i] < min_d &&
                net->nodes[i].type != NODE_PRUNED) {
                min_d = dist[i]; u = i;
            }
        }
        if (u == UINT32_MAX || u == to) break;
        seen[u] = true;

        SRHNNode4 *node = &net->nodes[u];
        for (uint8_t k = 0; k < node->n_neighbors; k++) {
            uint32_t v = node->neighbors[k];
            if (v >= net->n_nodes || seen[v]) continue;
            if (net->nodes[v].type == NODE_PRUNED) continue;

            EdgeType et = node->edge_types ? node->edge_types[k] : EDGE_ASSOC;
            /* Cost: causal edges are cheaper, inhibitory more expensive */
            float cost = 1.f - srhn4_temporal_weight(node->edge_weights[k], node->edge_timestamps ? node->edge_timestamps[k] : 0);
            if (et == EDGE_CAUSAL || et == EDGE_CAUSES)  cost *= 0.5f; /* prefer */
            if (et == EDGE_INHIBITORY) cost *= 2.f;                    /* avoid */
            if (et == EDGE_CONTRADICTS) cost = FLT_MAX / 2.f;          /* block */

            float nd = dist[u] + cost;
            if (nd < dist[v]) {
                dist[v] = nd;
                prev[v] = u;
            }
        }
    }

    /* Reconstruct path */
    if (prev[to] == UINT32_MAX && dist[to] == FLT_MAX) {
        free(dist); free(prev); free(seen);
        return; /* No path found */
    }

    uint32_t path[MAX_CHAIN_LEN];
    int path_len = 0;
    uint32_t cur = to;
    while (cur != UINT32_MAX && path_len < MAX_CHAIN_LEN) {
        path[path_len++] = cur;
        cur = prev[cur];
    }

    /* Reverse path */
    for (int i = 0; i < path_len / 2; i++) {
        uint32_t tmp = path[i]; path[i] = path[path_len-1-i]; path[path_len-1-i] = tmp;
    }

    out->length = (uint8_t)path_len;
    out->is_causal = true;
    float total = 1.f;
    for (int i = 0; i < path_len && i < MAX_CHAIN_LEN; i++) {
        out->nodes[i] = path[i];
        strncpy(out->labels[i], net->nodes[path[i]].label, 127);
        if (i > 0) {
            /* Find edge between path[i-1] and path[i] */
            SRHNNode4 *prev_node = &net->nodes[path[i-1]];
            for (uint8_t k = 0; k < prev_node->n_neighbors; k++) {
                if (prev_node->neighbors[k] == path[i]) {
                    out->strengths[i] = srhn4_temporal_weight(prev_node->edge_weights[k], prev_node->edge_timestamps ? prev_node->edge_timestamps[k] : 0);
                    if (prev_node->edge_types)
                        out->edge_types[i] = prev_node->edge_types[k];
                    total *= srhn4_temporal_weight(prev_node->edge_weights[k], prev_node->edge_timestamps ? prev_node->edge_timestamps[k] : 0);
                    break;
                }
            }
        } else {
            out->strengths[0] = 1.f;
        }
    }
    out->total_strength = total;

    free(dist); free(prev); free(seen);
}

/* ── Contradiction detection ─────────────────────────────────── */
void srhn4_find_contradictions(SRHNNetwork4 *net, SRHNResult4 *result) {
    result->n_contradictions = 0;

    /* Check all pairs of highly-activated nodes */
    for (uint32_t i = 0; i < result->n_activated && result->n_contradictions < 4; i++) {
        uint32_t na = result->activated_nodes[i];
        if (result->activations[i] < 0.25f) continue;
        if (na >= net->n_nodes) continue;

        SRHNNode4 *node_a = &net->nodes[na];
        for (uint8_t k = 0; k < node_a->n_neighbors; k++) {
            EdgeType et = node_a->edge_types ? node_a->edge_types[k] : EDGE_ASSOC;
            if (et != EDGE_CONTRADICTS && et != EDGE_ANTONYM) continue;

            uint32_t nb = node_a->neighbors[k];
            if (nb >= net->n_nodes) continue;

            /* Check if nb is also activated */
            for (uint32_t j = 0; j < result->n_activated; j++) {
                if (result->activated_nodes[j] == nb && result->activations[j] > 0.25f) {
                    Contradiction *c = &result->contradictions[result->n_contradictions++];
                    c->node_a = na; c->node_b = nb;
                    c->confidence = (result->activations[i] + result->activations[j]) * 0.5f
                                    * node_a->edge_weights[k];
                    strncpy(c->label_a, node_a->label, 127);
                    strncpy(c->label_b, net->nodes[nb].label, 127);
                    break;
                }
            }
        }
    }
    if (result->n_contradictions > 0)
        result->n_contradictions > 0 /* already set */; (void)0;
}

/* ── Hyperedge evaluation ────────────────────────────────────── */
void srhn4_eval_hyperedges(SRHNNetwork4 *net, SRHNResult4 *result) {
    for (uint32_t e = 0; e < net->n_edges; e++) {
        SRHNEdge4 *edge = &net->edges[e];
        if (edge->n_nodes == 0) continue;  /* dead edge */

        float min_act = 1.f, sum_act = 0.f;
        int   act_cnt = 0;

        for (uint8_t i = 0; i < edge->n_nodes; i++) {
            uint32_t nid = edge->nodes[i];
            if (nid >= net->n_nodes) continue;
            float act = net->nodes[nid].activation;
            if (act > 0.08f) act_cnt++;
            if (act < min_act) min_act = act;
            sum_act += act;
        }

        float avg = edge->n_nodes > 0 ? sum_act / edge->n_nodes : 0.f;

        if (edge->is_causal) {
            /* All nodes must be active for causal edge to fire */
            edge->activation = (act_cnt == edge->n_nodes)
                                ? min_act * edge->causal_strength : 0.f;
        } else {
            edge->activation = avg * edge->weight;
        }

        if (edge->activation > 0.2f && result->n_fired < 255) {
            result->fired_edges[result->n_fired++] = e;
            edge->fire_count++;
            edge->last_fired_us = srhn4_timestamp_us();
        }
    }
}
