/*
 * srhn_core4.c  —  SRHN v4 Core Network + Integrated Research Pipeline
 *
 * The v4 query pipeline extends v3's 19 steps with 8 research modules:
 *
 *  Steps 1–7:   [unchanged from v3] lang/abbrev/embed/blend/decay/wm/HNSW
 *  Step 7.5:   [R8] Spectral seed re-scoring
 *  Step 8:     [v3] BFS propagation (now uses temporal effective weights)
 *  Step 9:     [R3] Hyperedge attention (replaces binary hyperedge eval)
 *  Step 9.5:   [R2] Distributional message passing
 *  Steps 10–19: [unchanged] chains/contradictions/concepts/calib/grow/LLM/ctx/episode/maint/metrics
 *  Step 19.5:  [R7] PAC-Bayes bound applied to result
 *  Periodic:   [R4] AE quality filter every AE_RUN_INTERVAL queries
 *              [R8] Spectral recompute every SPECTRAL_RECOMPUTE queries
 */
#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>
#include <strings.h>
/* Safe size multiply (also defined in srhn_scale.h but inlined here for zero-dep) */
static inline bool safe_mul_sz(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > ((size_t)-1) / a) return false;
    *out = a * b; return true;
}

#include <math.h>
#include <stdio.h>
#include <strings.h>

#define PAGERANK_INTERVAL 100

/* ── Network lifecycle ──────────────────────────────────────── */

SRHNNetwork4 *srhn4_create(void) {
    SRHNNetwork4 *net = (SRHNNetwork4 *)calloc(1, sizeof(SRHNNetwork4));
    if (!net) return NULL;

    net->node_cap = SRHN4_INIT_NODES;
    net->edge_cap = SRHN4_INIT_EDGES;
    net->nodes    = (SRHNNode4 *)calloc(net->node_cap, sizeof(SRHNNode4));
    net->edges    = (SRHNEdge4 *)calloc(net->edge_cap, sizeof(SRHNEdge4));
    if (!net->nodes || !net->edges) { free(net->nodes); free(net->edges); free(net); return NULL; }

    if (!srhn4_hnsw_init(&net->hnsw, SRHN4_INIT_NODES)) {
        free(net->nodes); free(net->edges); free(net); return NULL;
    }

    net->lexicon.cap     = 4096;
    net->lexicon.entries = (LexEntry *)calloc(net->lexicon.cap, sizeof(LexEntry));
    net->physics_laws    = (PhysicsLaw *)calloc(256, sizeof(PhysicsLaw));

    srhn4_episodic_init(net, EPISODIC_CAP);

    /* Init v4 research modules */
    srhn4_ae_init (&net->autoenc, 0.001f);
    srhn4_vae_init(&net->vae,     0.001f);
    srhn4_msgpass_init(&net->msgpass, SRHN4_INIT_NODES, SRHN4_INIT_EDGES);
    srhn4_spectral_init(&net->spectral, SRHN4_INIT_NODES);
    memset(&net->pacbayes, 0, sizeof(net->pacbayes));
    net->pacbayes.delta = PACBAYES_DELTA;

    net->rng_state        = srhn4_timestamp_us() ^ 0xDEADBEEFCAFEULL;
    net->prune_interval   = 25.f;
    net->selfgrow_enabled = true;
    net->selfgrow_thresh  = 0.18f;
    net->pagerank.damping = 0.85f;
    net->last_error       = SRHN_OK;

    /* Enable all research modules by default */
    net->use_multisem       = true;
    net->use_msgpass        = true;
    net->use_hyperedge_attn = true;
    net->use_autoenc        = true;
    net->use_temporal       = true;
    net->use_vae            = false;  /* off: needs explicit user activation */
    net->use_pacbayes       = true;
    net->use_spectral       = true;

    pthread_rwlock_init(&net->nodes_lock, NULL);
    pthread_rwlock_init(&net->edges_lock, NULL);
    pthread_mutex_init (&net->query_lock, NULL);

    return net;
}

void srhn4_destroy(SRHNNetwork4 *net) {
    if (!net) return;
    srhn4_wal_close(net);
    pthread_rwlock_wrlock(&net->nodes_lock);
    for (uint32_t i = 0; i < net->n_nodes; i++) {
        free(net->nodes[i].neighbors);
        free(net->nodes[i].edge_weights);
        free(net->nodes[i].edge_types);
        free(net->nodes[i].edge_timestamps);
    }
    pthread_rwlock_unlock(&net->nodes_lock);
    pthread_rwlock_wrlock(&net->edges_lock);
    for (uint32_t i = 0; i < net->n_edges; i++) {
        free(net->edges[i].nodes);
        free(net->edges[i].attn_scores);
    }
    pthread_rwlock_unlock(&net->edges_lock);
    srhn4_hnsw_free(&net->hnsw);
    srhn4_episodic_free(net);
    srhn4_llm_free(&net->llm);
    srhn4_msgpass_free(&net->msgpass);
    srhn4_spectral_free(&net->spectral);
    free(net->lexicon.entries);
    free(net->nodes); free(net->edges);
    free(net->physics_laws);
    free(net->pagerank.scores);
    pthread_rwlock_destroy(&net->nodes_lock);
    pthread_rwlock_destroy(&net->edges_lock);
    pthread_mutex_destroy (&net->query_lock);
    free(net);
}

void srhn4_reset(SRHNNetwork4 *net) {
    if (!net) return;
    pthread_rwlock_wrlock(&net->nodes_lock);
    for (uint32_t i = 0; i < net->n_nodes; i++) {
        free(net->nodes[i].neighbors); free(net->nodes[i].edge_weights);
        free(net->nodes[i].edge_types); free(net->nodes[i].edge_timestamps);
    }
    memset(net->nodes, 0, net->node_cap * sizeof(SRHNNode4));
    net->n_nodes = 0;
    pthread_rwlock_unlock(&net->nodes_lock);
    pthread_rwlock_wrlock(&net->edges_lock);
    for (uint32_t i = 0; i < net->n_edges; i++) {
        free(net->edges[i].nodes); free(net->edges[i].attn_scores);
    }
    memset(net->edges, 0, net->edge_cap * sizeof(SRHNEdge4));
    net->n_edges = 0;
    pthread_rwlock_unlock(&net->edges_lock);
    srhn4_hnsw_free(&net->hnsw);
    srhn4_hnsw_init(&net->hnsw, SRHN4_INIT_NODES);
    free(net->lexicon.entries);
    net->lexicon.entries   = (LexEntry *)calloc(4096, sizeof(LexEntry));
    net->lexicon.n_entries = 0;
    net->lexicon.cap       = 4096;
    srhn4_episodic_free(net);
    srhn4_episodic_init(net, EPISODIC_CAP);
    srhn4_working_mem_clear(net);
    memset(&net->stats,   0, sizeof(net->stats));
    memset(&net->context, 0, sizeof(net->context));
    memset(&net->calib,   0, sizeof(net->calib));
    net->global_time = 0.f; net->query_counter = 0; net->selfgrow_count = 0;
    net->n_physics_laws = 0;
}

/* ── Growth helpers ─────────────────────────────────────────── */

static bool grow_nodes(SRHNNetwork4 *net) {
    if (net->node_cap >= SRHN4_MAX_NODES) return false;
    uint32_t nc = (net->node_cap * 2 > SRHN4_MAX_NODES) ? SRHN4_MAX_NODES : net->node_cap * 2;
    size_t new_bytes; if (!safe_mul_sz(nc, sizeof(SRHNNode4), &new_bytes)) return false;
    SRHNNode4 *p = (SRHNNode4 *)realloc(net->nodes, new_bytes);
    if (!p) return false;
    memset(p + net->node_cap, 0, (nc - net->node_cap) * sizeof(SRHNNode4));
    net->nodes = p; net->node_cap = nc; return true;
}
static bool grow_edges(SRHNNetwork4 *net) {
    if (net->edge_cap >= SRHN4_MAX_EDGES) return false;
    uint32_t nc = (net->edge_cap * 2 > SRHN4_MAX_EDGES) ? SRHN4_MAX_EDGES : net->edge_cap * 2;
    size_t new_bytes; if (!safe_mul_sz(nc, sizeof(SRHNEdge4), &new_bytes)) return false;
    SRHNEdge4 *p = (SRHNEdge4 *)realloc(net->edges, new_bytes);
    if (!p) return false;
    memset(p + net->edge_cap, 0, (nc - net->edge_cap) * sizeof(SRHNEdge4));
    net->edges = p; net->edge_cap = nc; return true;
}

/* ── Node management ─────────────────────────────────────────── */

uint32_t srhn4_add_node(SRHNNetwork4 *net, const char *label, NodeType type, LangID lang) {
    pthread_rwlock_wrlock(&net->nodes_lock);
    if (net->n_nodes >= net->node_cap && !grow_nodes(net)) {
        net->last_error = SRHN_ERR_FULL; pthread_rwlock_unlock(&net->nodes_lock);
        return UINT32_MAX;
    }
    uint32_t id     = net->n_nodes++;
    SRHNNode4 *node  = &net->nodes[id];
    memset(node, 0, sizeof(SRHNNode4));
    node->id = id; node->type = type; node->lang = lang;
    strncpy(node->label, label, 127); node->label[127] = '\0';
    node->entropy_score = 0.5f; node->proto_id = id;
    node->lex_id = UINT32_MAX; node->hnsw_layer = -1;
    node->in_fast_ring = (net->stats.fast_ring_count < FAST_RING_CAP / 4);
    node->neighbor_cap = 8;
    node->neighbors      = (uint32_t  *)malloc(8 * sizeof(uint32_t));
    node->edge_weights   = (float     *)malloc(8 * sizeof(float));
    node->edge_types     = (EdgeType  *)malloc(8 * sizeof(EdgeType));
    node->edge_timestamps= (uint64_t  *)calloc(8, sizeof(uint64_t));
    if (!node->neighbors || !node->edge_weights || !node->edge_types || !node->edge_timestamps) {
        free(node->neighbors); free(node->edge_weights);
        free(node->edge_types); free(node->edge_timestamps);
        node->neighbors = NULL; net->n_nodes--;
        net->last_error = SRHN_ERR_OOM;
        pthread_rwlock_unlock(&net->nodes_lock); return UINT32_MAX;
    }

    /* Compute embedding */
    srhn4_embed_text(net, label, lang, &node->sig);

    /* [R1] Compute orthogonal views from the embedding */
    if (net->use_multisem) srhn4_orthogonalize_views(&node->sig);

    net->stats.total_nodes++;
    if ((int)lang < LANG_COUNT) net->stats.nodes_by_lang[(int)lang]++;
    if (node->in_fast_ring) net->stats.fast_ring_count++;
    pthread_rwlock_unlock(&net->nodes_lock);

    srhn4_hnsw_insert(&net->hnsw, id, &node->sig, net);
    srhn4_wal_write(net, WAL_ADD_NODE, id, (uint32_t)type, 0.f,
                     label, (uint32_t)(strlen(label)+1));
    return id;
}

uint32_t srhn4_add_concept(SRHNNetwork4 *net, const char *text) {
    return srhn4_add_node(net, text, NODE_CONCEPT, srhn4_detect_lang(text));
}

void srhn4_connect(SRHNNetwork4 *net, uint32_t a, uint32_t b, float weight, EdgeType type) {
    if (a >= net->n_nodes || b >= net->n_nodes || a == b) return;
    weight = srhn4_clampf(weight, 0.001f, 1.f);
    pthread_rwlock_wrlock(&net->nodes_lock);
    SRHNNode4 *na = &net->nodes[a], *nb = &net->nodes[b];
    for (uint8_t i = 0; i < na->n_neighbors; i++) {
        if (na->neighbors[i] == b) {
            na->edge_weights[i] = weight;
            if (na->edge_types) na->edge_types[i] = type;
            pthread_rwlock_unlock(&net->nodes_lock); return;
        }
    }
    /* Grow if needed */
    if (na->n_neighbors >= na->neighbor_cap) {
        uint16_t nc = (uint16_t)(na->neighbor_cap >= MAX_NEIGHBORS ? MAX_NEIGHBORS : na->neighbor_cap*2);
        uint32_t  *tmp_n  = (uint32_t *)realloc(na->neighbors,      nc*sizeof(uint32_t));
        float     *tmp_w  = (float    *)realloc(na->edge_weights,   nc*sizeof(float));
        EdgeType  *tmp_t  = (EdgeType *)realloc(na->edge_types,     nc*sizeof(EdgeType));
        uint64_t  *tmp_ts = (uint64_t *)realloc(na->edge_timestamps,nc*sizeof(uint64_t));
        if (!tmp_n || !tmp_w || !tmp_t) {
            free(tmp_n); free(tmp_w); free(tmp_t); free(tmp_ts);
            pthread_rwlock_unlock(&net->nodes_lock); return;
        }
        na->neighbors = tmp_n; na->edge_weights = tmp_w;
        na->edge_types = tmp_t;
        if (tmp_ts) na->edge_timestamps = tmp_ts;
        else if (!na->edge_timestamps) na->edge_timestamps = (uint64_t *)calloc(nc, sizeof(uint64_t));
        na->neighbor_cap = nc;
    }
    if (nb->n_neighbors >= nb->neighbor_cap) {
        uint16_t nc = (uint16_t)(nb->neighbor_cap >= MAX_NEIGHBORS ? MAX_NEIGHBORS : nb->neighbor_cap*2);
        uint32_t  *tmp_n  = (uint32_t *)realloc(nb->neighbors,      nc*sizeof(uint32_t));
        float     *tmp_w  = (float    *)realloc(nb->edge_weights,   nc*sizeof(float));
        EdgeType  *tmp_t  = (EdgeType *)realloc(nb->edge_types,     nc*sizeof(EdgeType));
        uint64_t  *tmp_ts = (uint64_t *)realloc(nb->edge_timestamps,nc*sizeof(uint64_t));
        if (!tmp_n || !tmp_w || !tmp_t) {
            free(tmp_n); free(tmp_w); free(tmp_t); free(tmp_ts);
            pthread_rwlock_unlock(&net->nodes_lock); return;
        }
        nb->neighbors = tmp_n; nb->edge_weights = tmp_w;
        nb->edge_types = tmp_t;
        if (tmp_ts) nb->edge_timestamps = tmp_ts;
        else if (!nb->edge_timestamps) nb->edge_timestamps = (uint64_t *)calloc(nc, sizeof(uint64_t));
        nb->neighbor_cap = nc;
    }
    if (na->n_neighbors < MAX_NEIGHBORS) {
        na->neighbors   [na->n_neighbors] = b;
        na->edge_weights[na->n_neighbors] = weight;
        if (na->edge_types)      na->edge_types     [na->n_neighbors] = type;
        if (na->edge_timestamps) na->edge_timestamps[na->n_neighbors] = 0;
        na->n_neighbors++;
    }
    if (nb->n_neighbors < MAX_NEIGHBORS) {
        nb->neighbors   [nb->n_neighbors] = a;
        nb->edge_weights[nb->n_neighbors] = weight;
        if (nb->edge_types)      nb->edge_types     [nb->n_neighbors] = type;
        if (nb->edge_timestamps) nb->edge_timestamps[nb->n_neighbors] = 0;
        nb->n_neighbors++;
    }
    pthread_rwlock_unlock(&net->nodes_lock);
    srhn4_wal_write(net, WAL_ADD_EDGE, a, b, weight, &type, sizeof(EdgeType));
}

uint32_t srhn4_add_edge(SRHNNetwork4 *net, uint32_t *nids, uint8_t n,
                         float strength, EdgeType type, const char *relation) {
    if (!nids || n < 2) return UINT32_MAX;
    pthread_rwlock_wrlock(&net->edges_lock);
    if (net->n_edges >= net->edge_cap && !grow_edges(net)) {
        net->last_error = SRHN_ERR_FULL; pthread_rwlock_unlock(&net->edges_lock);
        return UINT32_MAX;
    }
    if (n > MAX_EDGE_NODES) n = MAX_EDGE_NODES;
    uint32_t id  = net->n_edges++;
    SRHNEdge4 *e = &net->edges[id];
    memset(e, 0, sizeof(SRHNEdge4));
    e->id = id; e->n_nodes = n; e->weight = strength;
    e->causal_strength = strength; e->type = type;
    e->is_causal = (type == EDGE_CAUSAL || type == EDGE_CAUSES);
    strncpy(e->relation, relation ? relation : "", 127);
    e->nodes = (uint32_t *)malloc(n * sizeof(uint32_t));
    e->attn_scores = (float *)calloc(n, sizeof(float)); /* [R3] */
    if (!e->nodes) {
        net->n_edges--; net->last_error = SRHN_ERR_OOM;
        pthread_rwlock_unlock(&net->edges_lock); return UINT32_MAX;
    }
    memcpy(e->nodes, nids, n * sizeof(uint32_t));
    net->stats.total_edges++;
    pthread_rwlock_unlock(&net->edges_lock);
    return id;
}

/* ── Pruning ─────────────────────────────────────────────────── */

void srhn4_prune_pass(SRHNNetwork4 *net) {
    uint32_t pruned = 0;
    pthread_rwlock_wrlock(&net->nodes_lock);
    for (uint32_t i = 0; i < net->n_nodes; i++) {
        SRHNNode4 *node = &net->nodes[i];
        if (node->type == NODE_PRUNED || node->in_fast_ring) continue;
        if (node->auto_created && node->usage_count < 0.5f &&
            net->global_time - node->last_active > 10.f) {
            node->type = NODE_PRUNED; pruned++; net->stats.pruned_nodes++;
            continue;
        }
        float usage_rate  = node->usage_count / (net->global_time + 1.f);
        float recency     = expf(-(net->global_time - node->last_active) * 0.06f);
        node->entropy_score = srhn4_clampf(
            0.5f*usage_rate + 0.3f*recency + 0.2f*node->importance, 0.f, 1.f);
        if (node->entropy_score < PRUNE_THRESH && node->usage_count < 3.f) {
            float best_r = 0.f; uint32_t best_nb = UINT32_MAX;
            for (uint8_t k = 0; k < node->n_neighbors; k++) {
                uint32_t nb = node->neighbors[k];
                if (nb >= net->n_nodes || net->nodes[nb].type == NODE_PRUNED) continue;
                /* [R1] Use multi-sem resonance for better prune decisions */
                float r = net->use_multisem
                    ? srhn4_multisem_resonance(&node->sig, &net->nodes[nb].sig)
                    : srhn4_resonance(&node->sig, &net->nodes[nb].sig);
                if (r > best_r) { best_r = r; best_nb = nb; }
            }
            if (best_nb != UINT32_MAX && best_r > 0.40f) {
                srhn4_sig_blend(&net->nodes[best_nb].sig, &node->sig, 0.12f);
                if (net->use_multisem) srhn4_orthogonalize_views(&net->nodes[best_nb].sig);
                net->nodes[best_nb].usage_count += node->usage_count * 0.35f;
                node->type = NODE_PRUNED; pruned++; net->stats.pruned_nodes++;
            }
        }
    }
    pthread_rwlock_unlock(&net->nodes_lock);
    pthread_rwlock_wrlock(&net->edges_lock);
    for (uint32_t e = 0; e < net->n_edges; e++) {
        SRHNEdge4 *edge = &net->edges[e];
        if (edge->n_nodes > 0 && edge->fire_count == 0 && net->global_time > 60.f) {
            free(edge->nodes); free(edge->attn_scores);
            edge->nodes = NULL; edge->attn_scores = NULL; edge->n_nodes = 0;
        }
    }
    pthread_rwlock_unlock(&net->edges_lock);
    if (pruned > 0) net->stats.total_prunes += pruned;
}

void srhn4_promote_fast(SRHNNetwork4 *net, uint32_t nid) {
    if (nid >= net->n_nodes || net->nodes[nid].in_fast_ring) return;
    if (net->stats.fast_ring_count >= FAST_RING_CAP) {
        float min_s = 2.f; uint32_t evict = UINT32_MAX;
        for (uint32_t i = 0; i < net->n_nodes; i++) {
            if (!net->nodes[i].in_fast_ring) continue;
            float s = net->nodes[i].entropy_score*0.5f + net->nodes[i].importance*0.5f;
            if (s < min_s) { min_s = s; evict = i; }
        }
        if (evict != UINT32_MAX) { net->nodes[evict].in_fast_ring = false; net->stats.fast_ring_count--; }
    }
    net->nodes[nid].in_fast_ring = true; net->stats.fast_ring_count++;
}

/* ── Main Query Pipeline ─────────────────────────────────────── */

SRHNResult4 srhn4_query(SRHNNetwork4 *net, const char *input) {
    SRHNResult4 result;
    memset(&result, 0, sizeof(result));
    uint64_t t_start = srhn4_timestamp_us();
    net->global_time += 1.f; net->stats.total_queries++; net->query_counter++;
    if (!input || !*input) { snprintf(result.response, RESP_BUF_SIZE, "Empty query."); return result; }

    /* 1. Detect language */
    LangID input_lang = srhn4_detect_lang(input);
    result.response_lang = input_lang;

    /* 2. Abbreviation expansion */
    char expanded[1024];
    strncpy(expanded, input, 1023); expanded[1023] = '\0';
    {
        char tokens[32][MAX_WORD_LEN]; int nt = srhn4_tok_simple(input, tokens, 32);
        char rebuilt[1024] = "";
        for (int t = 0; t < nt; t++) {
            for (uint32_t li = 0; li < net->lexicon.n_entries; li++) {
                LexEntry *le = &net->lexicon.entries[li];
                for (uint8_t a = 0; a < le->n_abbrevs; a++)
                    if (strcasecmp(le->abbrevs[a], tokens[t]) == 0)
                        { strncpy(tokens[t], le->word, MAX_WORD_LEN-1); break; }
            }
            if (t > 0) strncat(rebuilt, " ", 1023-strlen(rebuilt));
            strncat(rebuilt, tokens[t], 1023-strlen(rebuilt));
        }
        if (strlen(rebuilt) > 0) strncpy(expanded, rebuilt, 1023);
    }

    /* 3. Query embedding */
    Signature qsig;
    srhn4_embed_text(net, expanded, input_lang, &qsig);
    if (net->use_multisem) srhn4_orthogonalize_views(&qsig);

    /* 4. Cross-turn context blending */
    Signature effective_sig;
    srhn4_ctx_blend_query(net, &effective_sig, &qsig);
    if (net->use_multisem) srhn4_orthogonalize_views(&effective_sig);

    /* 5. Decay activations */
    pthread_rwlock_wrlock(&net->nodes_lock);
    for (uint32_t i = 0; i < net->n_nodes; i++) {
        net->nodes[i].activation *= 0.20f;
        if (net->nodes[i].activation < 0.01f) { net->nodes[i].activation = 0.f; net->nodes[i].is_active = false; }
    }
    pthread_rwlock_unlock(&net->nodes_lock);

    /* 6. Working memory decay */
    srhn4_working_mem_apply(net);

    /* 7. HNSW seed search */
    uint32_t seed_ids[16]; float seed_dists[16];
    int n_seeds = srhn4_hnsw_search(&net->hnsw, &effective_sig,
                                     net->hnsw.ef_search, seed_ids, seed_dists, 16, net);
    if (n_seeds > 0) {
        net->stats.hnsw_hits++;
        /* [R8] Spectral re-scoring of seeds */
        if (net->use_spectral && net->spectral.valid)
            srhn4_spectral_rescore_seeds(net, seed_ids, seed_dists, n_seeds);
    } else {
        net->stats.hnsw_misses++;
        /* Linear fallback */
        float best_r[16] = {0}; int nb = 0;
        pthread_rwlock_rdlock(&net->nodes_lock);
        for (uint32_t i = 0; i < net->n_nodes; i++) {
            if (net->nodes[i].type == NODE_PRUNED) continue;
            /* [R1] Use multi-sem resonance for seed finding */
            float r = net->use_multisem
                ? srhn4_multisem_resonance(&effective_sig, &net->nodes[i].sig)
                : srhn4_resonance(&effective_sig, &net->nodes[i].sig);
            if (net->nodes[i].in_fast_ring) r = srhn4_clampf(r*1.15f, 0.f, 1.f);
            if (r >= RESONANCE_THRESH && nb < 16) {
                seed_ids[nb]=i; seed_dists[nb]=1.f-r; best_r[nb]=r; nb++;
            }
        }
        pthread_rwlock_unlock(&net->nodes_lock);
        n_seeds = nb;
    }

    /* 8. BFS propagation (uses temporal effective weights via srhn4_propagate4) */
    float best_res = 0.f; result.seeds_used = (uint32_t)n_seeds;
    for (int s = 0; s < n_seeds && result.n_activated < MAX_ACTIVATED-64; s++) {
        float r = 1.f - seed_dists[s];
        if (r < RESONANCE_THRESH) continue;
        if (r > best_res) best_res = r;
        if (seed_ids[s] < net->n_nodes && net->nodes[seed_ids[s]].in_fast_ring)
            r = srhn4_clampf(r*1.15f, 0.f, 1.f);
        srhn4_propagate(net, seed_ids[s], r, &result);
    }
    result.best_resonance = best_res;

    /* 9. [R3] Hyperedge attention (replaces binary eval) */
    if (net->use_hyperedge_attn)
        srhn4_hyperedge_attn_run(net, &result);
    else
        srhn4_eval_hyperedges(net, &result);

    /* 9.5 [R2] Distributional message passing */
    if (net->use_msgpass) srhn4_msgpass_run(net, &result);

    /* 10. Reasoning chains */
    srhn4_build_chains(net, &result);

    /* 11. Contradiction detection */
    srhn4_find_contradictions(net, &result);

    /* 12. Top concepts */
    result.n_top_concepts = 0;
    typedef struct { uint32_t id; float act; } TopNode;
    TopNode top8[8]; int ntop = 0;
    for (uint32_t i = 0; i < result.n_activated; i++) {
        uint32_t nid = result.activated_nodes[i]; float act = result.activations[i];
        if (nid >= net->n_nodes || net->nodes[nid].type == NODE_PRUNED) continue;
        int j = ntop;
        while (j > 0 && top8[j-1].act < act && j < 8) { top8[j]=top8[j-1]; j--; }
        if (j < 8) { top8[j]=(TopNode){nid,act}; if (ntop<8) ntop++; }
    }
    for (int i = 0; i < ntop && result.n_top_concepts < 8; i++)
        strncpy(result.top_concepts[result.n_top_concepts++], net->nodes[top8[i].id].label, 127);

    /* 13. Calibrated confidence */
    float raw_conf = srhn4_clampf(
        best_res*0.65f + (result.n_activated>0?0.25f:0.f) + (result.n_chains>0?0.10f:0.f), 0.f, 1.f);
    result.confidence = srhn4_calibrated_conf(net, raw_conf);

    /* 14. Self-grow */
    if (best_res < net->selfgrow_thresh) srhn4_selfgrow_from_query(net, expanded, best_res);

    /* 15. LLM answer */
    pthread_mutex_lock(&net->query_lock);
    srhn4_answer(net, input, &result);
    pthread_mutex_unlock(&net->query_lock);

    /* 16. Context update */
    srhn4_ctx_update(net, input, &result, &qsig);

    /* 17. Episodic store */
    srhn4_episodic_store(net, &result, input, 0.f);

    /* 18. Periodic maintenance */
    if (net->query_counter % 25 == 0) srhn4_prune_pass(net);
    if (net->query_counter % PAGERANK_INTERVAL == 0) srhn4_run_pagerank(net);
    if (net->use_temporal && net->query_counter % 50 == 0) srhn4_temporal_decay_pass(net);
    if (net->use_autoenc && net->query_counter % AE_RUN_INTERVAL == 0) srhn4_ae_run(net);
    if (net->use_spectral && net->query_counter % SPECTRAL_RECOMPUTE == 0) srhn4_spectral_compute(net);
    if (net->use_pacbayes && net->query_counter % 200 == 0) srhn4_pacbayes_update(net);

    /* 19. Metrics */
    uint64_t t_end = srhn4_timestamp_us();
    result.latency_ms   = (double)(t_end - t_start) / 1000.0;
    result.compute_used = (float)result.n_activated / (float)(net->n_nodes+1);
    result.hops_taken   = result.n_activated;
    net->stats.avg_latency_ms = 0.93*net->stats.avg_latency_ms + 0.07*result.latency_ms;
    net->stats.sparsity       = result.compute_used;
    net->stats.active_nodes   = result.n_activated;
    if (result.used_llm) net->stats.llm_calls++;

    /* 19.5 [R7] PAC-Bayes bounds — always apply (safe defaults before first update) */
    if (net->use_pacbayes)
        srhn4_pacbayes_apply_to_result(net, &result);

    return result;
}

/* ── Feedback ────────────────────────────────────────────────── */

void srhn4_feedback(SRHNNetwork4 *net, uint32_t query_id, float reward) {
    FeedbackEntry4 *entry = &net->feedback_buf[net->feedback_head % FEEDBACK_RING];
    entry->query_id  = query_id;
    entry->reward    = srhn4_clampf(reward, -1.f, 1.f);
    entry->timestamp_us = srhn4_timestamp_us();
    entry->n_active  = 0;
    pthread_rwlock_rdlock(&net->nodes_lock);
    for (uint32_t i = 0; i < net->n_nodes && entry->n_active < 128; i++)
        if (net->nodes[i].is_active) entry->active_nodes[entry->n_active++] = i;
    pthread_rwlock_unlock(&net->nodes_lock);
    if (net->episodic.n_episodes > 0) {
        uint32_t ep_idx = (net->episodic.head-1+net->episodic.cap) % net->episodic.cap;
        net->episodic.episodes[ep_idx].reward = reward;
    }
    net->feedback_head++;
    if (net->feedback_count < FEEDBACK_RING) net->feedback_count++;
    srhn4_apply_feedback(net);

    /* [R1] Update multi-semantic view weights from feedback */
    if (net->use_multisem && net->context.count > 0) {
        uint8_t prev = (net->context.head-1+CTX_HISTORY) % CTX_HISTORY;
        for (uint8_t i = 0; i < entry->n_active; i++) {
            uint32_t nid = entry->active_nodes[i];
            if (nid >= net->n_nodes) continue;
            srhn4_multisem_update_weights(&net->nodes[nid].sig,
                &net->context.turns[prev].blended_sig, reward, 0.02f);
        }
    }
}

/* ── Stats ───────────────────────────────────────────────────── */

SRHNStats4 srhn4_get_stats(SRHNNetwork4 *net) {
    float esum=0.f, isum=0.f; uint32_t live=0, fast=0;
    pthread_rwlock_rdlock(&net->nodes_lock);
    for (uint32_t i = 0; i < net->n_nodes; i++) {
        if (net->nodes[i].type == NODE_PRUNED) continue;
        live++; esum+=net->nodes[i].entropy_score; isum+=net->nodes[i].importance;
        if (net->nodes[i].in_fast_ring) fast++;
    }
    pthread_rwlock_unlock(&net->nodes_lock);
    net->stats.total_nodes=live; net->stats.fast_ring_count=fast;
    net->stats.avg_entropy=live>0?esum/live:0.f;
    net->stats.avg_pagerank=live>0?isum/live:0.f;
    return net->stats;
}

void srhn4_print_stats(SRHNNetwork4 *net) {
    SRHNStats4 s = srhn4_get_stats(net);
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║       SRHN v4 Statistics (Research Edition)      ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║ Nodes:        %8u  (auto: %6u)         ║\n", s.total_nodes, s.auto_nodes);
    printf("║ Edges:        %8u                          ║\n", s.total_edges);
    printf("║ Fast ring:    %8u                          ║\n", s.fast_ring_count);
    printf("║ Avg PageRank: %12.6f                     ║\n", s.avg_pagerank);
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║ Queries:      %8llu  latency: %6.1fms       ║\n", (unsigned long long)s.total_queries, s.avg_latency_ms);
    printf("║ HNSW hits:    %8u  misses:  %8u        ║\n", s.hnsw_hits, s.hnsw_misses);
    printf("║ LLM calls:    %8u                          ║\n", s.llm_calls);
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║ [R4] AE runs:        %5u  promoted: %4u     ║\n", s.ae_runs, s.ae_nodes_promoted);
    printf("║ [R6] VAE cands:      %5u  accepted: %4u     ║\n", s.vae_candidates_generated, s.vae_candidates_accepted);
    printf("║ [R7] PAC-Bayes bound: %.4f                   ║\n", s.avg_pacbayes_bound);
    printf("║ [R8] Spectral runs:  %5u                     ║\n", s.spectral_runs);
    printf("║ [R2] Avg msg entropy: %.3f                   ║\n", s.avg_msg_entropy);
    printf("║ Feedback score:      %6.4f                   ║\n", s.feedback_score);
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║ Modules enabled:                                 ║\n");
    printf("║  R1:multisem=%s R2:msgpass=%s R3:attn=%s      ║\n",
           net->use_multisem?"Y":"N", net->use_msgpass?"Y":"N", net->use_hyperedge_attn?"Y":"N");
    printf("║  R4:ae=%s R5:temporal=%s R6:vae=%s           ║\n",
           net->use_autoenc?"Y":"N", net->use_temporal?"Y":"N", net->use_vae?"Y":"N");
    printf("║  R7:pacbayes=%s R8:spectral=%s               ║\n",
           net->use_pacbayes?"Y":"N", net->use_spectral?"Y":"N");
    printf("╚══════════════════════════════════════════════════╝\n\n");
}

const char *srhn4_error_str(SRHNError err) {
    static const char *msgs[] = {
        "OK","Null pointer","Out of memory","Out of bounds","Duplicate",
        "I/O error","Corrupt data","LLM error","Embed error","Full","Not found","Lock error"
    };
    return ((unsigned)err < 12) ? msgs[(int)err] : "Unknown";
}
