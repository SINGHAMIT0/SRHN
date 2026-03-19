/*
 * srhn_hyperedge_attn.c  —  [R3] Hyperedge Attention
 *
 * Replaces the binary "all-or-nothing" causal fire rule with softmax
 * attention scoring over hyperedge member nodes.
 *
 * For each hyperedge e with members {v_1, ..., v_k}:
 *   score(v_i, e) = activation(v_i) × edge_weight(v_i,e) × resonance(v_i, e_centroid)
 *   attn(v_i, e) = softmax(score(v_i, e) / T) over i
 *   edge_activation = Σ_i attn(v_i, e) × activation(v_i)
 *
 * This gives a graded activation that:
 *   - fires even when some members are weakly active
 *   - weights contributions by relevance to the hyperedge semantic centroid
 *   - preserves causal edges' directionality through the attention distribution
 */
#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>

#define ATTN_TEMPERATURE  0.8f

/*
 * Compute the centroid signature of a hyperedge's active member nodes.
 * Returns number of active members.
 */
static int edge_centroid(SRHNNetwork4 *net, SRHNEdge4 *edge, float *centroid) {
    memset(centroid, 0, EMBED_DIM * sizeof(float));
    int n_active = 0;
    for (uint8_t i = 0; i < edge->n_nodes; i++) {
        uint32_t nid = edge->nodes[i];
        if (nid >= net->n_nodes || net->nodes[nid].type == NODE_PRUNED) continue;
        float act = net->nodes[nid].activation;
        if (act < 0.05f) continue;
        for (int d = 0; d < EMBED_DIM; d++)
            centroid[d] += act * net->nodes[nid].sig.vec[d];
        n_active++;
    }
    if (n_active > 0) {
        float mag = 0.f;
        for (int d = 0; d < EMBED_DIM; d++) mag += centroid[d]*centroid[d];
        mag = sqrtf(mag) + 1e-8f;
        for (int d = 0; d < EMBED_DIM; d++) centroid[d] /= mag;
    }
    return n_active;
}

/*
 * Shannon entropy of the attention distribution.
 * Low entropy = one member dominates; high entropy = all members equal.
 */
float srhn4_edge_attn_entropy(SRHNEdge4 *edge) {
    if (!edge->attn_scores || edge->n_nodes == 0) return 0.f;
    float H = 0.f;
    for (uint8_t i = 0; i < edge->n_nodes; i++) {
        float p = edge->attn_scores[i];
        if (p > 1e-6f) H -= p * log2f(p);
    }
    return H;
}

/*
 * srhn4_hyperedge_attn_run — process all hyperedges with attention scoring.
 *
 * Called after BFS propagation, before reasoning chain building.
 * Updates edge->activation using attention-weighted member activations.
 * Records attn_entropy in each edge for downstream explainability.
 */
void srhn4_hyperedge_attn_run(SRHNNetwork4 *net, SRHNResult4 *result) {
    if (!net || !result) return;

    float   centroid[EMBED_DIM];
    float   scores  [MAX_EDGE_NODES];
    float   max_attn_overall = 0.f;

    for (uint32_t eid = 0; eid < net->n_edges; eid++) {
        SRHNEdge4 *edge = &net->edges[eid];
        if (!edge->nodes || edge->n_nodes < 2) continue;

        /* Compute edge centroid from active members */
        int n_active = edge_centroid(net, edge, centroid);
        if (n_active == 0) {
            edge->activation = 0.f;
            continue;
        }

        /* Allocate attention scores array if needed */
        if (!edge->attn_scores) {
            edge->attn_scores = (float *)calloc(edge->n_nodes, sizeof(float));
            if (!edge->attn_scores) continue;
        }

        /* Compute raw attention scores for each member */
        float score_max = -1e9f;
        for (uint8_t i = 0; i < edge->n_nodes && i < MAX_EDGE_NODES; i++) {
            uint32_t nid = edge->nodes[i];
            if (nid >= net->n_nodes || net->nodes[nid].type == NODE_PRUNED) {
                scores[i] = -1e9f; continue;
            }
            SRHNNode4 *node = &net->nodes[nid];
            float act = node->activation;
            float res = srhn4_dot(node->sig.vec, centroid, EMBED_DIM);
            /* score = activation × resonance-to-centroid */
            float s = act * srhn4_clampf(res, 0.f, 1.f);

            /* Edge-weight bonus: upweight strongly connected members */
            for (uint8_t k = 0; k < node->n_neighbors; k++) {
                bool found = false;
                for (uint8_t j = 0; j < edge->n_nodes; j++) {
                    if (node->neighbors[k] == edge->nodes[j]) { found = true; break; }
                }
                if (found) { s *= (1.f + node->edge_weights[k] * 0.3f); break; }
            }

            scores[i] = s / ATTN_TEMPERATURE;
            if (scores[i] > score_max) score_max = scores[i];
        }

        /* Softmax */
        float sum_exp = 0.f;
        for (uint8_t i = 0; i < edge->n_nodes && i < MAX_EDGE_NODES; i++) {
            if (scores[i] <= -1e8f) { edge->attn_scores[i] = 0.f; continue; }
            float e = expf(scores[i] - score_max);
            edge->attn_scores[i] = e;
            sum_exp += e;
        }
        if (sum_exp < 1e-8f) continue;
        for (uint8_t i = 0; i < edge->n_nodes && i < MAX_EDGE_NODES; i++)
            edge->attn_scores[i] /= sum_exp;

        /* Edge activation = attention-weighted sum of member activations */
        float attn_activation = 0.f;
        float max_single = 0.f;
        for (uint8_t i = 0; i < edge->n_nodes && i < MAX_EDGE_NODES; i++) {
            uint32_t nid = edge->nodes[i];
            if (nid >= net->n_nodes) continue;
            float contribution = edge->attn_scores[i] * net->nodes[nid].activation;
            attn_activation += contribution;
            if (edge->attn_scores[i] > max_single) max_single = edge->attn_scores[i];
        }

        /* Scale by edge base weight */
        edge->activation = srhn4_clampf(attn_activation * edge->weight, 0.f, 1.f);
        edge->attn_entropy = srhn4_edge_attn_entropy(edge);

        if (max_single > max_attn_overall) max_attn_overall = max_single;

        /* Fire if threshold crossed */
        if (edge->activation > 0.15f && result->n_fired < 255) {
            result->fired_edges[result->n_fired++] = eid;
            edge->fire_count++;
            edge->last_fired_us = srhn4_timestamp_us();
        }
    }

    result->max_attn_score = max_attn_overall;
}
