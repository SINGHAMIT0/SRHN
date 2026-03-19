/*
 * srhn_hnsw.c  —  Hierarchical Navigable Small World graph index
 *
 * Provides O(log n) approximate nearest-neighbor search over node signatures.
 * Replaces the O(n) linear scan in v2's query loop.
 *
 * Implementation based on: Malkov & Yashunin 2018 (arXiv:1603.09320)
 * Parameters: M=16, M0=32, ef_construction=200, ef_search=64
 */

#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── Priority queue (min-heap by distance) ─────────────────── */
typedef struct {
    uint32_t id;
    float    dist;   /* distance = 1 - cosine_similarity */
} PQItem;

typedef struct {
    PQItem  *data;
    int      n;
    int      cap;
    bool     is_max; /* true = max-heap, false = min-heap */
} PQ;

static PQ *pq_create(int cap, bool is_max) {
    PQ *q = (PQ *)malloc(sizeof(PQ));
    if (!q) return NULL;
    q->data   = (PQItem *)malloc(cap * sizeof(PQItem));
    q->n      = 0;
    q->cap    = cap;
    q->is_max = is_max;
    if (!q->data) { free(q); return NULL; }
    return q;
}

static void pq_free(PQ *q) { if (q) { free(q->data); free(q); } }

static void pq_push(PQ *q, uint32_t id, float dist) {
    if (q->n >= q->cap) {
        /* Grow */
        int nc = q->cap * 2;
        PQItem *p = (PQItem *)realloc(q->data, nc * sizeof(PQItem));
        if (!p) return;
        q->data = p; q->cap = nc;
    }
    int i = q->n++;
    q->data[i] = (PQItem){id, dist};
    /* Bubble up */
    while (i > 0) {
        int parent = (i - 1) / 2;
        bool swap = q->is_max ? (q->data[i].dist > q->data[parent].dist)
                               : (q->data[i].dist < q->data[parent].dist);
        if (!swap) break;
        PQItem tmp = q->data[i]; q->data[i] = q->data[parent]; q->data[parent] = tmp;
        i = parent;
    }
}

static PQItem pq_pop(PQ *q) {
    PQItem top = q->data[0];
    q->data[0] = q->data[--q->n];
    int i = 0;
    while (true) {
        int l = 2*i+1, r = 2*i+2, best = i;
        if (l < q->n) {
            bool lt = q->is_max ? (q->data[l].dist > q->data[best].dist)
                                 : (q->data[l].dist < q->data[best].dist);
            if (lt) best = l;
        }
        if (r < q->n) {
            bool rt = q->is_max ? (q->data[r].dist > q->data[best].dist)
                                 : (q->data[r].dist < q->data[best].dist);
            if (rt) best = r;
        }
        if (best == i) break;
        PQItem tmp = q->data[i]; q->data[i] = q->data[best]; q->data[best] = tmp;
        i = best;
    }
    return top;
}

static float pq_top_dist(PQ *q) { return q->n > 0 ? q->data[0].dist : FLT_MAX; }

/* ── Distance function (1 - cosine similarity) ──────────────── */
static float hnsw_dist(const Signature *a, const Signature *b) {
    float sim = srhn4_resonance(a, b);
    return 1.f - sim;
}

/* ── Layer height sampling ──────────────────────────────────── */
static int sample_layer(SRHNNetwork4 *net) {
    /* P(layer >= l) = (1/M)^l   → geometric distribution */
    int layer = 0;
    float p = 1.f / (float)HNSW_M;
    while (srhn4_randf(net) < p && layer < HNSW_MAX_LAYERS - 1)
        layer++;
    return layer;
}

/* ── Search layer ───────────────────────────────────────────── */
static void search_layer(HNSWIndex *idx, const Signature *query,
                          uint32_t ep_id, uint32_t ef, int layer,
                          SRHNNetwork4 *net, PQ *result) {
    bool *visited = (bool *)calloc(idx->n_nodes, sizeof(bool));
    if (!visited) return;

    PQ *candidates = pq_create(ef * 4 + 16, false); /* min-heap by dist */
    if (!candidates) { free(visited); return; }

    const Signature *ep_sig = &net->nodes[idx->nodes[ep_id].node_id].sig;
    float d = hnsw_dist(query, ep_sig);

    pq_push(candidates, ep_id, d);
    pq_push(result, ep_id, d);
    visited[ep_id] = true;

    while (candidates->n > 0) {
        PQItem cur = pq_pop(candidates);

        /* Early termination: if nearest candidate is farther than worst in result */
        if (result->n >= (int)ef && cur.dist > pq_top_dist(result))
            break;

        HNSWNode *cur_node = &idx->nodes[cur.id];
        int lc = (layer < cur_node->max_layer) ? layer : cur_node->max_layer;
        if (lc < 0) lc = 0;

        for (uint8_t i = 0; i < cur_node->n_links[lc]; i++) {
            uint32_t nb_hnsw_id = cur_node->links[lc][i];
            if (nb_hnsw_id >= idx->n_nodes || visited[nb_hnsw_id]) continue;
            visited[nb_hnsw_id] = true;

            uint32_t nb_node_id = idx->nodes[nb_hnsw_id].node_id;
            if (nb_node_id >= net->n_nodes) continue;

            const Signature *nb_sig = &net->nodes[nb_node_id].sig;
            float nd = hnsw_dist(query, nb_sig);

            if (result->n < (int)ef || nd < pq_top_dist(result)) {
                pq_push(candidates, nb_hnsw_id, nd);
                pq_push(result, nb_hnsw_id, nd);
                if (result->n > (int)ef) pq_pop(result);
            }
        }
    }

    pq_free(candidates);
    free(visited);
}

/* ── Select neighbors (heuristic) ──────────────────────────── */
static void select_neighbors(HNSWIndex *idx, const Signature *query,
                               PQ *candidates, int M,
                               uint32_t *out_links, uint8_t *out_n,
                               SRHNNetwork4 *net) {
    /* Simple: take top-M by distance */
    /* For heuristic: also prune redundant links */
    int sel = 0;
    while (candidates->n > 0 && sel < M) {
        PQItem item = pq_pop(candidates);
        uint32_t nid = idx->nodes[item.id].node_id;
        if (nid >= net->n_nodes) continue;

        /* Heuristic pruning: skip if closer to existing selected neighbor */
        bool dominated = false;
        for (int k = 0; k < sel; k++) {
            uint32_t sel_nid = idx->nodes[out_links[k]].node_id;
            if (sel_nid >= net->n_nodes) continue;
            float d_to_sel = hnsw_dist(&net->nodes[nid].sig,
                                        &net->nodes[sel_nid].sig);
            if (d_to_sel < item.dist) { dominated = true; break; }
        }
        if (!dominated)
            out_links[sel++] = item.id;
    }
    *out_n = (uint8_t)sel;
    (void)query; /* used implicitly via PQ distances */
}

/* ── Init / Free ────────────────────────────────────────────── */

bool srhn4_hnsw_init(HNSWIndex *idx, uint32_t init_cap) {
    if (!idx) return false;
    idx->nodes          = (HNSWNode *)calloc(init_cap, sizeof(HNSWNode));
    idx->n_nodes        = 0;
    idx->cap            = init_cap;
    idx->entry_point    = UINT32_MAX;
    idx->max_layer      = -1;
    idx->ef_construction = HNSW_EF_CONSTRUCTION;
    idx->ef_search      = HNSW_EF_SEARCH;

    if (!idx->nodes) return false;
    if (pthread_rwlock_init(&idx->lock, NULL) != 0) {
        free(idx->nodes); return false;
    }
    return true;
}

void srhn4_hnsw_free(HNSWIndex *idx) {
    if (!idx) return;
    for (uint32_t i = 0; i < idx->n_nodes; i++) {
        for (int l = 0; l < HNSW_MAX_LAYERS; l++)
            free(idx->nodes[i].links[l]);
    }
    free(idx->nodes);
    idx->nodes = NULL; idx->n_nodes = 0;
    pthread_rwlock_destroy(&idx->lock);
}

/* ── Insert ─────────────────────────────────────────────────── */
bool srhn4_hnsw_insert(HNSWIndex *idx, uint32_t node_id,
                        const Signature *sig, SRHNNetwork4 *net) {
    if (!idx || node_id >= net->n_nodes) return false;

    pthread_rwlock_wrlock(&idx->lock);

    /* Grow if needed */
    if (idx->n_nodes >= idx->cap) {
        uint32_t nc = idx->cap * 2;
        HNSWNode *np = (HNSWNode *)realloc(idx->nodes, nc * sizeof(HNSWNode));
        if (!np) { pthread_rwlock_unlock(&idx->lock); return false; }
        memset(np + idx->cap, 0, (nc - idx->cap) * sizeof(HNSWNode));
        idx->nodes = np; idx->cap = nc;
    }

    uint32_t hnsw_id = idx->n_nodes++;
    HNSWNode *new_node = &idx->nodes[hnsw_id];
    memset(new_node, 0, sizeof(HNSWNode));
    new_node->node_id  = node_id;
    new_node->max_layer = sample_layer(net);

    /* Store layer in the network node */
    net->nodes[node_id].hnsw_layer = (int8_t)new_node->max_layer;

    /* Allocate link arrays */
    for (int l = 0; l <= new_node->max_layer; l++) {
        int m = (l == 0) ? HNSW_M0 : HNSW_M;
        new_node->links[l]    = (uint32_t *)calloc(m * 2, sizeof(uint32_t));
        new_node->link_cap[l] = (uint8_t)(m * 2);
        new_node->n_links[l]  = 0;
    }

    /* First node: just set as entry point */
    if (idx->entry_point == UINT32_MAX) {
        idx->entry_point = hnsw_id;
        idx->max_layer   = new_node->max_layer;
        pthread_rwlock_unlock(&idx->lock);
        return true;
    }

    /* Greedy search from entry point down to layer+1 */
    uint32_t ep = idx->entry_point;
    int top_layer = idx->max_layer;

    for (int lc = top_layer; lc > new_node->max_layer; lc--) {
        PQ *result = pq_create(16, true);
        if (!result) break;
        search_layer(idx, sig, ep, 1, lc, net, result);
        if (result->n > 0) ep = result->data[0].id; /* closest at this layer */
        pq_free(result);
    }

    /* Insert at each layer from min(max_layer, new_node->max_layer) down to 0 */
    for (int lc = (new_node->max_layer < top_layer ? new_node->max_layer : top_layer);
         lc >= 0; lc--) {
        int m_layer = (lc == 0) ? HNSW_M0 : HNSW_M;

        PQ *candidates = pq_create(idx->ef_construction * 4 + 16, true);
        if (!candidates) break;

        search_layer(idx, sig, ep, idx->ef_construction, lc, net, candidates);

        /* Select M best neighbors */
        uint32_t selected[HNSW_M0 * 2];
        uint8_t  n_selected = 0;
        select_neighbors(idx, sig, candidates, m_layer,
                          selected, &n_selected, net);
        pq_free(candidates);

        /* Add bidirectional links */
        for (uint8_t i = 0; i < n_selected; i++) {
            uint32_t nb_hnsw = selected[i];

            /* new → neighbor */
            if (new_node->n_links[lc] < new_node->link_cap[lc])
                new_node->links[lc][new_node->n_links[lc]++] = nb_hnsw;

            /* neighbor → new */
            HNSWNode *nb_node = &idx->nodes[nb_hnsw];
            int nb_layer = lc < nb_node->max_layer ? lc : nb_node->max_layer;
            int nb_cap = (nb_layer == 0) ? HNSW_M0 : HNSW_M;

            if (nb_node->links[nb_layer] == NULL) {
                nb_node->links[nb_layer]    = (uint32_t *)calloc(nb_cap * 2, sizeof(uint32_t));
                nb_node->link_cap[nb_layer] = (uint8_t)(nb_cap * 2);
            }

            if (nb_node->n_links[nb_layer] < nb_node->link_cap[nb_layer]) {
                nb_node->links[nb_layer][nb_node->n_links[nb_layer]++] = hnsw_id;
            } else {
                /* Prune: replace farthest link if new is closer */
                float d_new = hnsw_dist(sig, &net->nodes[nb_node->node_id].sig);
                float max_d = 0.f; int max_i = -1;
                for (uint8_t k = 0; k < nb_node->n_links[nb_layer]; k++) {
                    uint32_t lnk = nb_node->links[nb_layer][k];
                    if (lnk >= idx->n_nodes) continue;
                    float d = hnsw_dist(&net->nodes[nb_node->node_id].sig,
                                         &net->nodes[idx->nodes[lnk].node_id].sig);
                    if (d > max_d) { max_d = d; max_i = k; }
                }
                if (max_i >= 0 && d_new < max_d)
                    nb_node->links[nb_layer][max_i] = hnsw_id;
            }
        }

        /* Update entry point for next layer */
        if (n_selected > 0) ep = selected[0];
    }

    /* Update global entry point if new node has higher layer */
    if (new_node->max_layer > idx->max_layer) {
        idx->entry_point = hnsw_id;
        idx->max_layer   = new_node->max_layer;
    }

    pthread_rwlock_unlock(&idx->lock);
    return true;
}

/* ── Search ─────────────────────────────────────────────────── */
int srhn4_hnsw_search(HNSWIndex *idx, const Signature *query,
                       uint32_t ef, uint32_t *result_ids, float *result_dists,
                       int k, SRHNNetwork4 *net) {
    if (!idx || idx->entry_point == UINT32_MAX || idx->n_nodes == 0) return 0;
    if (ef < (uint32_t)k) ef = (uint32_t)k;

    pthread_rwlock_rdlock(&idx->lock);

    uint32_t ep = idx->entry_point;

    /* Greedy descent to layer 1 */
    for (int lc = idx->max_layer; lc > 0; lc--) {
        PQ *w = pq_create(16, true);
        if (!w) break;
        search_layer(idx, query, ep, 1, lc, net, w);
        if (w->n > 0) ep = w->data[0].id;
        pq_free(w);
    }

    /* Search layer 0 with ef candidates */
    PQ *w = pq_create((int)ef * 4 + 16, true); /* max-heap */
    if (!w) { pthread_rwlock_unlock(&idx->lock); return 0; }

    search_layer(idx, query, ep, ef, 0, net, w);

    /* Extract top-k (w is max-heap, so sorted descending by dist) */
    /* Convert to min-heap order by draining into array */
    int n __attribute__((unused)) = w->n < k ? w->n : k;

    /* Collect all results sorted */
    PQItem *arr = (PQItem *)malloc(w->n * sizeof(PQItem));
    if (!arr) { pq_free(w); pthread_rwlock_unlock(&idx->lock); return 0; }
    int arr_n = 0;
    while (w->n > 0) arr[arr_n++] = pq_pop(w);

    /* arr is now sorted ascending by dist (min first after popping max-heap) */
    /* Actually for max-heap pops give largest first; reverse */
    /* Reverse to get ascending order */
    for (int i = 0; i < arr_n / 2; i++) {
        PQItem tmp = arr[i]; arr[i] = arr[arr_n-1-i]; arr[arr_n-1-i] = tmp;
    }

    int out = 0;
    for (int i = 0; i < arr_n && out < k; i++) {
        uint32_t node_id = idx->nodes[arr[i].id].node_id;
        if (node_id >= net->n_nodes) continue;
        if (net->nodes[node_id].type == NODE_PRUNED) continue;
        result_ids[out]   = node_id;
        result_dists[out] = arr[i].dist;
        out++;
    }

    free(arr);
    pq_free(w);
    pthread_rwlock_unlock(&idx->lock);
    return out;
}
