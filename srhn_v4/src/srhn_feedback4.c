/*
 * srhn_feedback.c  —  Learning & feedback for SRHN v3
 *
 * Implements:
 *   1. Hebbian edge weight learning — co-activation strengthens edges
 *   2. Contrastive signature updates — positive pulls, negative pushes
 *   3. PageRank — continuous node importance scoring
 *   4. Isotonic calibration — turn raw confidence into real probabilities
 */

#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Hebbian learning ───────────────────────────────────────── */
/*
 * Hebb's rule: "neurons that fire together wire together"
 * On positive reward: strengthen edges between co-active nodes
 * On negative reward: weaken those same edges
 *
 * Weight update: Δw = lr * reward * act_a * act_b
 * Bounded to [W_FLOOR, W_CEIL]
 */
#define HEBB_LR       0.04f
#define W_FLOOR       0.05f
#define W_CEIL        0.97f
#define HEBB_DECAY    0.9995f  /* very slow weight decay toward 0.5 */

void srhn4_hebbian_update(SRHNNetwork4 *net, FeedbackEntry4 *entry) {
    if (!entry || entry->n_active < 2) return;
    float reward = entry->reward;
    if (fabsf(reward) < 0.05f) return;

    float lr = HEBB_LR * fabsf(reward);

    for (uint8_t i = 0; i < entry->n_active; i++) {
        uint32_t na = entry->active_nodes[i];
        if (na >= net->n_nodes) continue;
        SRHNNode4 *node_a = &net->nodes[na];
        float act_a = node_a->activation + 0.1f; /* avoid zero */

        for (uint8_t j = 0; j < node_a->n_neighbors; j++) {
            uint32_t nb = node_a->neighbors[j];
            if (nb >= net->n_nodes) continue;

            /* Check if nb is also in active set */
            bool nb_active = false;
            for (uint8_t k = 0; k < entry->n_active; k++)
                if (entry->active_nodes[k] == nb) { nb_active = true; break; }
            if (!nb_active) continue;

            float act_b = net->nodes[nb].activation + 0.1f;
            float delta = lr * reward * act_a * act_b;

            /* Inhibitory edges: reverse Hebbian */
            EdgeType et = node_a->edge_types ? node_a->edge_types[j] : EDGE_ASSOC;
            if (et == EDGE_INHIBITORY) delta = -delta;

            /* Apply weight decay + update */
            float w = node_a->edge_weights[j];
            w = w * HEBB_DECAY + delta;
            w = srhn4_clampf(w, W_FLOOR, W_CEIL);
            node_a->edge_weights[j] = w;

            net->stats.hebbian_updates++;
        }

        /* Update causal edge strengths if applicable */
        if (reward > 0.f) {
            for (uint32_t e = 0; e < net->n_edges; e++) {
                SRHNEdge4 *edge = &net->edges[e];
                if (!edge->is_causal || edge->n_nodes == 0) continue;
                /* Check if both endpoints are in active set */
                bool all_active = true;
                for (uint8_t k = 0; k < edge->n_nodes; k++) {
                    bool found = false;
                    for (uint8_t m = 0; m < entry->n_active; m++)
                        if (entry->active_nodes[m] == edge->nodes[k]) { found = true; break; }
                    if (!found) { all_active = false; break; }
                }
                if (all_active)
                    edge->causal_strength = srhn4_clampf(
                        edge->causal_strength + lr * reward, W_FLOOR, W_CEIL);
            }
        }
    }
}

/* ── Contrastive signature update ──────────────────────────── */
/*
 * Positive feedback: pull active node signatures toward query signature
 * Negative feedback: push active node signatures away from query signature
 * Learning rate is proportional to |reward| × activation
 */
void srhn4_contrastive_sig(SRHNNetwork4 *net, FeedbackEntry4 *entry) {
    if (!entry || entry->n_active == 0) return;
    if (net->context.count == 0) return;

    float reward = entry->reward;
    if (fabsf(reward) < 0.1f) return;

    /* Get the query signature from the most recent context turn */
    uint8_t prev_idx = (net->context.head - 1 + CTX_HISTORY) % CTX_HISTORY;
    Signature *ctx_sig = &net->context.turns[prev_idx].blended_sig;

    float base_lr = 0.08f * fabsf(reward);

    for (uint8_t i = 0; i < entry->n_active; i++) {
        uint32_t nid = entry->active_nodes[i];
        if (nid >= net->n_nodes) continue;
        SRHNNode4 *node = &net->nodes[nid];

        float act = srhn4_clampf(node->activation, 0.f, 1.f);
        float lr  = base_lr * (0.3f + 0.7f * act); /* higher for more-active nodes */

        if (reward > 0.f) {
            /* Pull toward context */
            srhn4_sig_blend(&node->sig, ctx_sig, lr);
        } else {
            /* Push away from context */
            for (int d = 0; d < EMBED_DIM; d++) {
                node->sig.vec[d] -= lr * ctx_sig->vec[d];
            }
            /* Re-normalize */
            float mag = 0.f;
            for (int d = 0; d < EMBED_DIM; d++) mag += node->sig.vec[d] * node->sig.vec[d];
            mag = sqrtf(mag) + 1e-8f;
            for (int d = 0; d < EMBED_DIM; d++) node->sig.vec[d] /= mag;
            node->sig.magnitude = 1.f;
        }
    }
}

/* ── PageRank ────────────────────────────────────────────────── */
/*
 * Power iteration PageRank.
 * r(v) = (1-d)/N + d * Σ_{u→v} r(u)/out(u)
 *
 * Runs every PAGE_RANK_INTERVAL queries.
 * For large graphs (>50K nodes), uses approximate version with sampling.
 */
#define PAGERANK_ITERS    20
#define PAGERANK_DAMPING  0.85f
#define PAGERANK_INTERVAL 100

void srhn4_run_pagerank(SRHNNetwork4 *net) {
    uint32_t n = net->n_nodes;
    if (n == 0) return;

    /* Ensure scores array is allocated and sized */
    if (!net->pagerank.scores || net->pagerank.n_scores < n) {
        float *p = (float *)realloc(net->pagerank.scores, n * sizeof(float));
        if (!p) return;
        net->pagerank.scores   = p;
        net->pagerank.n_scores = n;
    }
    net->pagerank.damping = PAGERANK_DAMPING;

    float *rank     = net->pagerank.scores;
    float *rank_new = (float *)malloc(n * sizeof(float));
    if (!rank_new) return;

    float init = 1.f / (float)n;
    for (uint32_t i = 0; i < n; i++) rank[i] = init;

    for (int iter = 0; iter < PAGERANK_ITERS; iter++) {
        for (uint32_t i = 0; i < n; i++) rank_new[i] = (1.f - PAGERANK_DAMPING) / (float)n;

        for (uint32_t u = 0; u < n; u++) {
            SRHNNode4 *node = &net->nodes[u];
            if (node->type == NODE_PRUNED || node->n_neighbors == 0) continue;

            float share = PAGERANK_DAMPING * rank[u] / (float)node->n_neighbors;
            for (uint8_t k = 0; k < node->n_neighbors; k++) {
                uint32_t v = node->neighbors[k];
                if (v < n && net->nodes[v].type != NODE_PRUNED) {
                    rank_new[v] += share * node->edge_weights[k]; /* weight-aware */
                }
            }
        }

        /* Normalize */
        float sum = 0.f;
        for (uint32_t i = 0; i < n; i++) sum += rank_new[i];
        if (sum > 1e-8f)
            for (uint32_t i = 0; i < n; i++) rank_new[i] /= sum;

        /* Check convergence */
        float diff = 0.f;
        for (uint32_t i = 0; i < n; i++)
            diff += fabsf(rank_new[i] - rank[i]);

        memcpy(rank, rank_new, n * sizeof(float));
        if (diff < 1e-6f) break;
    }

    free(rank_new);

    /* Update node importance scores */
    float max_rank = 0.f, avg = 0.f; uint32_t live = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (net->nodes[i].type == NODE_PRUNED) continue;
        if (rank[i] > max_rank) max_rank = rank[i];
        avg += rank[i]; live++;
    }
    net->stats.avg_pagerank = live > 0 ? avg / live : 0.f;

    /* Normalize to [0, 1] and store in node */
    if (max_rank > 1e-8f) {
        for (uint32_t i = 0; i < n; i++) {
            if (net->nodes[i].type == NODE_PRUNED) continue;
            net->nodes[i].importance = rank[i] / max_rank;

            /* Auto-promote high-importance nodes to fast ring */
            if (net->nodes[i].importance > 0.6f && !net->nodes[i].in_fast_ring)
                srhn4_promote_fast(net, i);
        }
    }

    net->pagerank.last_run_query = net->query_counter;
    net->stats.pagerank_runs++;
    fprintf(stderr, "[pagerank] Run complete: avg=%.6f, max=%.6f, live=%u\n",
            avg / (live + 1), max_rank, live);
}

/* ── Isotonic calibration ────────────────────────────────────── */
/*
 * After enough feedback, map raw resonance scores to actual accuracy.
 * Uses isotonic regression (pool adjacent violators).
 */

void srhn4_calibrate(SRHNNetwork4 *net, float predicted_conf, bool was_correct) {
    CalibTable *ct = &net->calib;
    int bucket = (int)(predicted_conf * CALIB_BUCKETS);
    if (bucket >= CALIB_BUCKETS) bucket = CALIB_BUCKETS - 1;

    ct->bucket_sum[bucket] += predicted_conf;
    ct->bucket_pos[bucket] += was_correct ? 1.f : 0.f;
    ct->bucket_n  [bucket]++;

    /* Refit after accumulating enough samples */
    uint32_t total = 0;
    for (int i = 0; i < CALIB_BUCKETS; i++) total += ct->bucket_n[i];
    if (total < 50) return;

    /* Compute raw calibrated values */
    for (int i = 0; i < CALIB_BUCKETS; i++) {
        if (ct->bucket_n[i] > 0)
            ct->calibrated[i] = ct->bucket_pos[i] / ct->bucket_n[i];
        else
            ct->calibrated[i] = (float)i / CALIB_BUCKETS;
    }

    /* Pool adjacent violators (PAV) — correct O(n²) worst-case but bounded
       by CALIB_BUCKETS=20, so at most 400 iterations total */
    for (int pass = 0; pass < CALIB_BUCKETS; pass++) {
        bool any = false;
        for (int i = 0; i < CALIB_BUCKETS - 1; i++) {
            if (ct->calibrated[i] > ct->calibrated[i+1] + 1e-6f) {
                float w1 = (float)(ct->bucket_n[i]   + 1);
                float w2 = (float)(ct->bucket_n[i+1] + 1);
                float m  = (ct->calibrated[i] * w1 + ct->calibrated[i+1] * w2) / (w1 + w2);
                ct->calibrated[i]   = m;
                ct->calibrated[i+1] = m;
                any = true;
            }
        }
        if (!any) break;
    }

    ct->fitted = true;
}

float srhn4_calibrated_conf(SRHNNetwork4 *net, float raw_conf) {
    CalibTable *ct = &net->calib;
    if (!ct->fitted) return raw_conf;

    int bucket = (int)(raw_conf * CALIB_BUCKETS);
    if (bucket < 0) bucket = 0;
    if (bucket >= CALIB_BUCKETS) bucket = CALIB_BUCKETS - 1;

    /* Linear interpolation between buckets */
    if (bucket < CALIB_BUCKETS - 1) {
        float t = raw_conf * CALIB_BUCKETS - bucket;
        return ct->calibrated[bucket] * (1.f - t) + ct->calibrated[bucket + 1] * t;
    }
    return ct->calibrated[bucket];
}

/* ── Apply feedback (main entry point) ──────────────────────── */
void srhn4_apply_feedback(SRHNNetwork4 *net) {
    if (net->feedback_count == 0) return;

    uint32_t idx = (net->feedback_head - 1 + FEEDBACK_RING) % FEEDBACK_RING;
    FeedbackEntry4 *entry = &net->feedback_buf[idx];
    float reward = entry->reward;
    float lr     = 0.06f;

    /* 1. Hebbian weight learning */
    srhn4_hebbian_update(net, entry);

    /* 2. Contrastive signature learning */
    srhn4_contrastive_sig(net, entry);

    /* 3. Entropy / usage updates */
    for (uint8_t i = 0; i < entry->n_active; i++) {
        uint32_t nid = entry->active_nodes[i];
        if (nid >= net->n_nodes) continue;
        SRHNNode4 *node = &net->nodes[nid];

        if (reward > 0.f) {
            node->entropy_score = srhn4_clampf(node->entropy_score + lr * reward, 0.f, 1.f);
            node->usage_count  += reward * 2.f;
            if (node->entropy_score > 0.70f && !node->in_fast_ring)
                srhn4_promote_fast(net, nid);
        } else {
            node->entropy_score = srhn4_clampf(node->entropy_score + lr * reward, 0.f, 1.f);
        }
    }

    /* 4. Update calibration */
    bool was_correct = reward > 0.2f;
    if (net->context.count > 0) {
        uint8_t prev = (net->context.head - 1 + CTX_HISTORY) % CTX_HISTORY;
        srhn4_calibrate(net, net->context.turns[prev].confidence, was_correct);
    }

    /* 5. Rolling feedback score (window=16) */
    float total = 0.f;
    uint32_t cnt = net->feedback_count < 16 ? net->feedback_count : 16;
    for (uint32_t k = 0; k < cnt; k++) {
        uint32_t fi = (net->feedback_head - 1 - k + FEEDBACK_RING) % FEEDBACK_RING;
        total += net->feedback_buf[fi].reward;
    }
    net->stats.feedback_score = total / (float)(cnt + 1);

    /* 6. Trigger PageRank if interval elapsed */
    if (net->query_counter > 0 &&
        net->query_counter - net->pagerank.last_run_query >= PAGERANK_INTERVAL)
        srhn4_run_pagerank(net);
}
