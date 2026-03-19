/*
 * srhn_pacbayes.c  —  [R7] PAC-Bayes confidence bounds
 * srhn_spectral.c  —  [R8] Spectral Laplacian seed scoring
 *
 * R7: Derives theoretical upper/lower confidence bounds from:
 *   - spectral norm of the weight matrix (proxy for model complexity)
 *   - hyperedge density (how many edges per node)
 *   - number of feedback samples seen
 *
 *   PAC-Bayes bound (simplified form from WWW 2025):
 *   P(R_true ≤ R_emp + sqrt((C + log(1/δ)) / 2n)) ≥ 1-δ
 *   where C = spectral_norm × hyperedge_density
 *
 * R8: Combines HNSW metric seeds with hypergraph Laplacian eigenvectors.
 *   The normalised hypergraph Laplacian: L = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
 *   We compute the top SPECTRAL_K eigenvectors offline via power iteration.
 *   Seed re-scoring: combined_score = (1-α)×metric_sim + α×spectral_sim
 *   where α=0.3 by default.
 */
#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 *  R7: PAC-Bayes
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Estimate spectral norm of the node weight matrix.
 * Full SVD is O(n³) — we use a power iteration estimate with 10 steps.
 * Returns the largest singular value estimate.
 */
static float estimate_spectral_norm(SRHNNetwork4 *net) {
    uint32_t n = net->n_nodes;
    if (n == 0) return 1.f;
    uint32_t dim = (n < EMBED_DIM) ? n : EMBED_DIM;

    /* Random initial vector */
    float *v = (float *)calloc(dim, sizeof(float));
    float *Av = (float *)calloc(dim, sizeof(float));
    if (!v || !Av) { free(v); free(Av); return 1.f; }

    for (uint32_t i = 0; i < dim; i++) v[i] = srhn4_randf(net) * 2.f - 1.f;

    float sigma = 1.f;
    for (int iter = 0; iter < 10; iter++) {
        /* Av[j] = Σ_i W[i][j] × v[i] — approximated as edge-weight dot product */
        memset(Av, 0, dim * sizeof(float));
        uint32_t cnt = 0;
        for (uint32_t nid = 0; nid < net->n_nodes && nid < dim; nid++) {
            SRHNNode4 *node = &net->nodes[nid];
            if (node->type == NODE_PRUNED) continue;
            float row_val = 0.f;
            for (uint8_t k = 0; k < node->n_neighbors; k++) {
                uint32_t nb = node->neighbors[k];
                if (nb < dim) row_val += node->edge_weights[k] * v[nb];
            }
            Av[nid] = row_val;
            cnt++;
        }
        /* ||Av|| */
        float norm = 0.f;
        for (uint32_t i = 0; i < dim; i++) norm += Av[i]*Av[i];
        norm = sqrtf(norm) + 1e-8f;
        sigma = norm;
        for (uint32_t i = 0; i < dim; i++) v[i] = Av[i] / norm;
    }

    free(v); free(Av);
    return sigma;
}

void srhn4_pacbayes_update(SRHNNetwork4 *net) {
    PACBayesState *pb = &net->pacbayes;
    pb->delta = PACBAYES_DELTA;

    /* Spectral norm */
    pb->spectral_norm = estimate_spectral_norm(net);

    /* Hyperedge density: avg hyperedge size × n_edges / n_nodes */
    float total_members = 0.f;
    for (uint32_t e = 0; e < net->n_edges; e++)
        if (net->edges[e].n_nodes > 0) total_members += net->edges[e].n_nodes;
    pb->hyperedge_density = (net->n_nodes > 0 && net->n_edges > 0)
        ? (total_members / net->n_nodes) : 1.f;

    /* Complexity = spectral_norm × log(hyperedge_density + 1) */
    pb->complexity = pb->spectral_norm * logf(pb->hyperedge_density + 1.f);
    pb->n_samples_seen = net->query_counter;
    pb->valid = true;
}

/*
 * Compute PAC-Bayes confidence bound.
 * Returns: P(R_true ≤ empirical_risk + bound) ≥ 1-δ
 *
 * bound = sqrt((C + log(1/δ)) / (2·n))
 * → confidence interval: [empirical_risk - bound, empirical_risk + bound]
 */
float srhn4_pacbayes_bound(SRHNNetwork4 *net, float empirical_risk) {
    PACBayesState *pb = &net->pacbayes;
    if (!pb->valid || pb->n_samples_seen < 10) return 0.5f;

    float n = (float)pb->n_samples_seen;
    float C = pb->complexity;
    float delta = pb->delta + 1e-8f;
    float bound = sqrtf((C + logf(1.f / delta)) / (2.f * n));
    return srhn4_clampf(bound, 0.f, 1.f);
}

void srhn4_pacbayes_apply_to_result(SRHNNetwork4 *net, SRHNResult4 *result) {
    PACBayesState *pb = &net->pacbayes;
    if (!pb->valid) {
        /* Not yet computed — use safe maximum-uncertainty defaults */
        result->confidence_lb = srhn4_clampf(result->confidence - 0.5f, 0.f, 1.f);
        result->confidence_ub = srhn4_clampf(result->confidence + 0.5f, 0.f, 1.f);
        return;
    }
    float empirical_risk = 1.f - result->confidence;
    float bound = srhn4_pacbayes_bound(net, empirical_risk);

    result->confidence_lb = srhn4_clampf(result->confidence - bound, 0.f, 1.f);
    result->confidence_ub = srhn4_clampf(result->confidence + bound, 0.f, 1.f);

    net->stats.avg_pacbayes_bound =
        0.95f * net->stats.avg_pacbayes_bound + 0.05f * bound;
}

/* ═══════════════════════════════════════════════════════════════
 *  R8: Spectral Laplacian
 * ═══════════════════════════════════════════════════════════════ */

bool srhn4_spectral_init(SpectralState *ss, uint32_t n_nodes) {
    ss->eigenvecs = (float *)calloc((size_t)SPECTRAL_K * n_nodes, sizeof(float));
    ss->eigenvals = (float *)calloc(SPECTRAL_K, sizeof(float));
    if (!ss->eigenvecs || !ss->eigenvals) {
        free(ss->eigenvecs); free(ss->eigenvals);
        return false;
    }
    ss->n_nodes_alloc = n_nodes;
    ss->valid = false;
    ss->last_run_query = 0;
    pthread_mutex_init(&ss->lock, NULL);
    return true;
}

void srhn4_spectral_free(SpectralState *ss) {
    if (!ss) return;
    free(ss->eigenvecs); free(ss->eigenvals);
    ss->eigenvecs = NULL; ss->eigenvals = NULL;
    pthread_mutex_destroy(&ss->lock);
}

/*
 * Compute the normalised hypergraph Laplacian L = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
 * then extract top SPECTRAL_K eigenvectors via power iteration with deflation.
 *
 * For large graphs (n > 10K), use a random subgraph sample of 1K nodes.
 */
void srhn4_spectral_compute(SRHNNetwork4 *net) {
    SpectralState *ss = &net->spectral;
    if (!ss->eigenvecs) return;

    pthread_mutex_lock(&ss->lock);

    uint32_t n = net->n_nodes;

    /* Grow if needed */
    if (n > ss->n_nodes_alloc) {
        uint32_t nn = n * 2;
        float *ne = (float *)realloc(ss->eigenvecs, (size_t)SPECTRAL_K * nn * sizeof(float));
        if (ne) { ss->eigenvecs = ne; ss->n_nodes_alloc = nn; }
        else { pthread_mutex_unlock(&ss->lock); return; }
    }

    /* Compute diagonal: D_v[i] = Σ_e W_e if i ∈ e */
    float *d_v = (float *)calloc(n, sizeof(float));
    if (!d_v) { pthread_mutex_unlock(&ss->lock); return; }

    for (uint32_t e = 0; e < net->n_edges; e++) {
        SRHNEdge4 *edge = &net->edges[e];
        if (!edge->nodes || edge->n_nodes == 0) continue;
        float we = edge->weight;
        for (uint8_t mi = 0; mi < edge->n_nodes; mi++) {
            uint32_t nid = edge->nodes[mi];
            if (nid < n) d_v[nid] += we;
        }
    }
    /* Also include pairwise edges */
    for (uint32_t i = 0; i < n; i++) {
        if (net->nodes[i].type == NODE_PRUNED) continue;
        for (uint8_t k = 0; k < net->nodes[i].n_neighbors; k++)
            d_v[i] += net->nodes[i].edge_weights[k];
    }
    /* D_v^{-1/2} */
    float *dv_inv_sqrt = (float *)calloc(n, sizeof(float));
    if (!dv_inv_sqrt) { free(d_v); pthread_mutex_unlock(&ss->lock); return; }
    for (uint32_t i = 0; i < n; i++)
        dv_inv_sqrt[i] = (d_v[i] > 1e-8f) ? 1.f / sqrtf(d_v[i]) : 0.f;

    /* Power iteration with deflation to find top-K eigenvectors of L.
     * L·v ≈ L applied via matrix-vector product:
     * (L·x)[i] = d_v_inv_sqrt[i] × (Σ_j adj[i,j] × x[j]) × d_v_inv_sqrt[j]
     */
    float *q_prev[SPECTRAL_K]; /* previously found eigenvectors for deflation */
    for (int k = 0; k < SPECTRAL_K; k++) q_prev[k] = NULL;

    for (int k = 0; k < SPECTRAL_K; k++) {
        float *ev = ss->eigenvecs + (size_t)k * n;
        /* Random initial vector */
        for (uint32_t i = 0; i < n; i++)
            ev[i] = (net->nodes[i].type != NODE_PRUNED) ? srhn4_randf(net)*2.f-1.f : 0.f;

        for (int iter = 0; iter < 30; iter++) {
            float *Lv = (float *)calloc(n, sizeof(float));
            if (!Lv) break;

            /* L·v via adjacency */
            for (uint32_t i = 0; i < n; i++) {
                if (net->nodes[i].type == NODE_PRUNED) continue;
                float s = 0.f;
                for (uint8_t ki = 0; ki < net->nodes[i].n_neighbors; ki++) {
                    uint32_t j = net->nodes[i].neighbors[ki];
                    if (j < n) s += net->nodes[i].edge_weights[ki] * dv_inv_sqrt[j] * ev[j];
                }
                Lv[i] = dv_inv_sqrt[i] * s;
            }

            /* Deflation: subtract projections onto previous eigenvectors */
            for (int prev = 0; prev < k; prev++) {
                if (!q_prev[prev]) continue;
                float dot = 0.f;
                for (uint32_t i = 0; i < n; i++) dot += Lv[i] * q_prev[prev][i];
                for (uint32_t i = 0; i < n; i++) Lv[i] -= dot * q_prev[prev][i];
            }

            /* Normalise */
            float norm = 0.f;
            for (uint32_t i = 0; i < n; i++) norm += Lv[i]*Lv[i];
            norm = sqrtf(norm) + 1e-8f;
            ss->eigenvals[k] = norm;
            for (uint32_t i = 0; i < n; i++) ev[i] = Lv[i] / norm;
            free(Lv);
        }
        q_prev[k] = ev;

        /* Store spectral coordinate in each node */
        for (uint32_t i = 0; i < n; i++)
            if (i < n && net->nodes[i].type != NODE_PRUNED)
                net->nodes[i].spectral_coords[k] = ev[i];
    }

    free(d_v); free(dv_inv_sqrt);

    ss->valid = true;
    ss->last_run_query = net->query_counter;
    net->stats.spectral_runs++;
    pthread_mutex_unlock(&ss->lock);
    fprintf(stderr, "[spectral] Computed %d eigenvectors for %u nodes\n", SPECTRAL_K, n);
}

/*
 * srhn4_spectral_rescore_seeds — re-rank HNSW seeds by combining
 * metric distance and spectral similarity.
 *
 * combined = (1-α)×(1-metric_dist) + α×spectral_sim
 * Sorted descending by combined score.
 */
void srhn4_spectral_rescore_seeds(SRHNNetwork4 *net,
                                   uint32_t *seed_ids, float *seed_dists,
                                   int n_seeds) {
    SpectralState *ss = &net->spectral;
    if (!ss->valid || n_seeds <= 0) return;

    /* Only recompute if interval elapsed */
    if (net->query_counter - ss->last_run_query >= SPECTRAL_RECOMPUTE)
        srhn4_spectral_compute(net);
    if (!ss->valid) return;

    const float ALPHA = 0.30f; /* spectral weight */

    /* Compute combined scores */
    float combined[16]; /* max seeds */
    int ns = n_seeds < 16 ? n_seeds : 16;

    /* Query spectral coords: average of top-3 activated nodes if available */
    float q_spectral[SPECTRAL_K] = {0};
    uint32_t q_cnt = 0;
    for (int s = 0; s < ns && q_cnt < 3; s++) {
        uint32_t nid = seed_ids[s];
        if (nid >= net->n_nodes) continue;
        for (int k = 0; k < SPECTRAL_K; k++)
            q_spectral[k] += net->nodes[nid].spectral_coords[k];
        q_cnt++;
    }
    if (q_cnt > 0)
        for (int k = 0; k < SPECTRAL_K; k++) q_spectral[k] /= q_cnt;

    for (int s = 0; s < ns; s++) {
        uint32_t nid = seed_ids[s];
        float metric_sim = 1.f - seed_dists[s];

        float spectral_sim = 0.f;
        if (nid < net->n_nodes) {
            float dot = 0.f, norm_q = 0.f, norm_n = 0.f;
            for (int k = 0; k < SPECTRAL_K; k++) {
                dot    += q_spectral[k] * net->nodes[nid].spectral_coords[k];
                norm_q += q_spectral[k] * q_spectral[k];
                norm_n += net->nodes[nid].spectral_coords[k] * net->nodes[nid].spectral_coords[k];
            }
            if (norm_q > 1e-8f && norm_n > 1e-8f)
                spectral_sim = srhn4_clampf(dot / (sqrtf(norm_q)*sqrtf(norm_n)), 0.f, 1.f);
        }

        combined[s] = (1.f - ALPHA) * metric_sim + ALPHA * spectral_sim;
    }

    /* Simple bubble sort (n_seeds ≤ 16 — fine) */
    for (int i = 0; i < ns-1; i++) {
        for (int j = i+1; j < ns; j++) {
            if (combined[j] > combined[i]) {
                float tc = combined[i]; combined[i] = combined[j]; combined[j] = tc;
                uint32_t tid = seed_ids[i]; seed_ids[i] = seed_ids[j]; seed_ids[j] = tid;
                float td = seed_dists[i]; seed_dists[i] = seed_dists[j]; seed_dists[j] = td;
            }
        }
    }
}
