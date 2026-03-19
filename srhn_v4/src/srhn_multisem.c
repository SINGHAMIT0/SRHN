/*
 * srhn_multisem.c  —  [R1] Multi-semantic orthogonal views
 *                     [R2] Distributional message passing
 *
 * R1: MSH-FSL insight: each concept has N_VIEWS=3 independent semantic
 *     facets. We decompose the 384-dim embedding into 3×128-dim
 *     sub-vectors via Gram-Schmidt orthogonalisation.
 *     srhn4_multisem_resonance() computes a weighted combination of
 *     per-view cosine similarities, enabling "force (physics)" and
 *     "force (emotion)" to co-exist without collision.
 *
 * R2: Distributional message passing replaces scalar spike-decay BFS
 *     with attention-weighted bidirectional signature blending.
 *     node → hyperedge: aggregate member node signatures with softmax weights
 *     hyperedge → node: distribute hyperedge aggregate back to members
 *     After 2 rounds the node signatures carry neighbourhood context.
 */
#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>

/* ── R1: Multi-semantic orthogonal views ─────────────────────── */

/*
 * Gram-Schmidt orthogonalisation of 3 VIEW_DIM-dim sub-vectors.
 * Input: sig->vec[384] is already unit-normalised.
 * Output: sig->views[3][128] are mutually orthogonal unit vectors.
 *         sig->view_weights[3] are initialised to 1/3.
 *
 * Strategy:
 *   view[0] = first 128 dims (projected from full embedding)
 *   view[1] = middle 128 dims, made orthogonal to view[0]
 *   view[2] = last 128 dims, made orthogonal to view[0] and view[1]
 */
void srhn4_orthogonalize_views(Signature *sig) {
    /* Extract 3 raw sub-vectors */
    float *v0 = sig->views[0];
    float *v1 = sig->views[1];
    float *v2 = sig->views[2];

    /* Copy raw segments from full embedding */
    memcpy(v0, sig->vec,                VIEW_DIM * sizeof(float));
    memcpy(v1, sig->vec + VIEW_DIM,     VIEW_DIM * sizeof(float));
    memcpy(v2, sig->vec + 2*VIEW_DIM,   VIEW_DIM * sizeof(float));

    /* Gram-Schmidt: normalise v0 */
    float norm = 0.f;
    for (int i = 0; i < VIEW_DIM; i++) norm += v0[i]*v0[i];
    norm = sqrtf(norm) + 1e-8f;
    for (int i = 0; i < VIEW_DIM; i++) v0[i] /= norm;

    /* Orthogonalise v1 against v0 */
    float proj01 = 0.f;
    for (int i = 0; i < VIEW_DIM; i++) proj01 += v1[i]*v0[i];
    for (int i = 0; i < VIEW_DIM; i++) v1[i] -= proj01 * v0[i];
    norm = 0.f;
    for (int i = 0; i < VIEW_DIM; i++) norm += v1[i]*v1[i];
    norm = sqrtf(norm) + 1e-8f;
    for (int i = 0; i < VIEW_DIM; i++) v1[i] /= norm;

    /* Orthogonalise v2 against v0 and v1 */
    float proj02 = 0.f, proj12 = 0.f;
    for (int i = 0; i < VIEW_DIM; i++) proj02 += v2[i]*v0[i];
    for (int i = 0; i < VIEW_DIM; i++) proj12 += v2[i]*v1[i];
    for (int i = 0; i < VIEW_DIM; i++) v2[i] -= proj02*v0[i] + proj12*v1[i];
    norm = 0.f;
    for (int i = 0; i < VIEW_DIM; i++) norm += v2[i]*v2[i];
    norm = sqrtf(norm) + 1e-8f;
    for (int i = 0; i < VIEW_DIM; i++) v2[i] /= norm;

    /* Uniform initial view weights */
    sig->view_weights[0] = 1.f / N_VIEWS;
    sig->view_weights[1] = 1.f / N_VIEWS;
    sig->view_weights[2] = 1.f / N_VIEWS;
}

/*
 * srhn4_multisem_resonance — weighted sum of per-view cosine similarities.
 *
 * For each view pair (a.views[k], b.views[k]):
 *   sim_k = cos(a.views[k], b.views[k])
 *
 * Combined = Σ_k sqrt(a.view_weights[k] * b.view_weights[k]) * sim_k
 *
 * The geometric mean of weights gives a symmetric measure. If one node
 * doesn't activate view k (weight≈0), that view contributes little.
 */
float srhn4_multisem_resonance(const Signature *a, const Signature *b) {
    if (!a || !b) return 0.f;

    /* Fast hash pre-filter (same as single-view) */
    uint32_t diff = (uint32_t)__builtin_popcount(a->hash ^ b->hash);
    if (diff > 28) return 0.f;

    float total_sim = 0.f;
    float total_weight = 0.f;

    for (int k = 0; k < N_VIEWS; k++) {
        float wa = a->view_weights[k];
        float wb = b->view_weights[k];
        float w  = sqrtf(wa * wb);
        if (w < 0.01f) continue;

        float dot = 0.f;
        for (int d = 0; d < VIEW_DIM; d++)
            dot += a->views[k][d] * b->views[k][d];

        total_sim    += w * srhn4_clampf(dot, -1.f, 1.f);
        total_weight += w;
    }

    if (total_weight < 1e-6f) return srhn4_resonance(a, b); /* fallback */

    float ms_sim = srhn4_clampf(total_sim / total_weight, 0.f, 1.f);

    /* Blend 70% multi-sem + 30% full-dim for stability */
    float full_sim = srhn4_resonance(a, b);
    return 0.70f * ms_sim + 0.30f * full_sim;
}

/*
 * Update view weights via contrastive learning signal.
 * On positive reward: increase weight of most-resonant view.
 * On negative reward: decrease weight of most-resonant view.
 */
void srhn4_multisem_update_weights(Signature *a, const Signature *b,
                                    float reward, float lr) {
    int best_k = 0;
    float best_sim = -2.f;
    for (int k = 0; k < N_VIEWS; k++) {
        float dot = 0.f;
        for (int d = 0; d < VIEW_DIM; d++)
            dot += a->views[k][d] * b->views[k][d];
        if (dot > best_sim) { best_sim = dot; best_k = k; }
    }
    a->view_weights[best_k] = srhn4_clampf(
        a->view_weights[best_k] + lr * reward, 0.05f, 0.90f);

    /* Re-normalise weights to sum to 1 */
    float sum = 0.f;
    for (int k = 0; k < N_VIEWS; k++) sum += a->view_weights[k];
    if (sum > 1e-6f)
        for (int k = 0; k < N_VIEWS; k++) a->view_weights[k] /= sum;
}

/* ── R2: Distributional Message Passing ──────────────────────── */

bool srhn4_msgpass_init(MsgPassState *mp, uint32_t n_nodes, uint32_t n_edges) {
    mp->n_nodes_alloc = n_nodes;
    mp->n_edges_alloc = n_edges;
    mp->temperature   = 1.0f;
    mp->enabled       = true;

    mp->node_msgs   = (float *)calloc((size_t)n_nodes * EMBED_DIM, sizeof(float));
    mp->edge_msgs   = (float *)calloc((size_t)n_edges * EMBED_DIM, sizeof(float));
    mp->attn_weights= (float *)calloc((size_t)n_nodes * n_edges,   sizeof(float));

    if (!mp->node_msgs || !mp->edge_msgs || !mp->attn_weights) {
        free(mp->node_msgs); free(mp->edge_msgs); free(mp->attn_weights);
        mp->enabled = false;
        return false;
    }
    return true;
}

void srhn4_msgpass_free(MsgPassState *mp) {
    if (!mp) return;
    free(mp->node_msgs); free(mp->edge_msgs); free(mp->attn_weights);
    mp->node_msgs = NULL; mp->edge_msgs = NULL; mp->attn_weights = NULL;
}

void srhn4_msgpass_resize(MsgPassState *mp, uint32_t new_nodes, uint32_t new_edges) {
    if (new_nodes <= mp->n_nodes_alloc && new_edges <= mp->n_edges_alloc) return;
    uint32_t nn = new_nodes > mp->n_nodes_alloc ? new_nodes * 2 : mp->n_nodes_alloc;
    uint32_t ne = new_edges > mp->n_edges_alloc ? new_edges * 2 : mp->n_edges_alloc;

    float *nm = (float *)realloc(mp->node_msgs,    (size_t)nn * EMBED_DIM * sizeof(float));
    float *em = (float *)realloc(mp->edge_msgs,    (size_t)ne * EMBED_DIM * sizeof(float));
    float *aw = (float *)realloc(mp->attn_weights, (size_t)nn * ne * sizeof(float));

    if (nm) { mp->node_msgs    = nm; mp->n_nodes_alloc = nn; }
    if (em) { mp->edge_msgs    = em; mp->n_edges_alloc = ne; }
    if (aw) { mp->attn_weights = aw; }
}

/*
 * srhn4_msgpass_run — one round of bidirectional distributional message passing.
 *
 * Phase 1 (node → hyperedge):
 *   For each active hyperedge e:
 *     score_i = exp(sim(node_i_sig, mean_sig) / T) for each member node i
 *     a_i = softmax(score_i)
 *     edge_msg[e] = Σ_i a_i × node_sig[i]
 *
 * Phase 2 (hyperedge → node):
 *   For each active node v:
 *     msg_v = mean over edges containing v of edge_msg[e] × edge_activation[e]
 *     new_sig[v] = 0.85 × old_sig[v] + 0.15 × norm(msg_v)
 *
 * This updates the node signatures in-place, blending local neighbourhood
 * context into each node — same idea as graph attention networks, but over
 * hyperedges rather than pairwise edges.
 */
void srhn4_msgpass_run(SRHNNetwork4 *net, SRHNResult4 *result) {
    MsgPassState *mp = &net->msgpass;
    if (!mp->enabled || !mp->node_msgs) return;
    if (result->n_fired == 0) return;

    srhn4_msgpass_resize(mp, net->n_nodes, net->n_edges);

    const float BLEND = 0.15f;
    const float T = mp->temperature;

    /* Phase 1: aggregate node sigs into fired hyperedge messages */
    for (uint32_t fi = 0; fi < result->n_fired; fi++) {
        uint32_t eid = result->fired_edges[fi];
        if (eid >= net->n_edges) continue;
        SRHNEdge4 *edge = &net->edges[eid];
        if (edge->n_nodes == 0) continue;

        /* Compute mean signature over active members */
        float mean_sig[EMBED_DIM] = {0};
        uint8_t n_active = 0;
        for (uint8_t mi = 0; mi < edge->n_nodes; mi++) {
            uint32_t nid = edge->nodes[mi];
            if (nid >= net->n_nodes || !net->nodes[nid].is_active) continue;
            for (int d = 0; d < EMBED_DIM; d++)
                mean_sig[d] += net->nodes[nid].sig.vec[d];
            n_active++;
        }
        if (n_active == 0) continue;
        float mscale = 1.f / n_active;
        for (int d = 0; d < EMBED_DIM; d++) mean_sig[d] *= mscale;

        /* Compute attention scores via softmax over member similarities */
        float scores[MAX_EDGE_NODES] = {0};
        float score_max = -1e9f;
        for (uint8_t mi = 0; mi < edge->n_nodes && mi < MAX_EDGE_NODES; mi++) {
            uint32_t nid = edge->nodes[mi];
            if (nid >= net->n_nodes) { scores[mi] = -1e9f; continue; }
            float dot = srhn4_dot(net->nodes[nid].sig.vec, mean_sig, EMBED_DIM);
            scores[mi] = dot / T;
            if (scores[mi] > score_max) score_max = scores[mi];
        }
        float sum_exp = 0.f;
        for (uint8_t mi = 0; mi < edge->n_nodes && mi < MAX_EDGE_NODES; mi++) {
            scores[mi] = expf(scores[mi] - score_max);
            sum_exp += scores[mi];
        }
        if (sum_exp < 1e-8f) continue;

        /* Update edge attention scores */
        if (!edge->attn_scores) {
            edge->attn_scores = (float *)calloc(edge->n_nodes, sizeof(float));
        }

        /* Build edge message: attention-weighted sum of node sigs */
        float *emsg = mp->edge_msgs + (size_t)eid * EMBED_DIM;
        memset(emsg, 0, EMBED_DIM * sizeof(float));
        for (uint8_t mi = 0; mi < edge->n_nodes && mi < MAX_EDGE_NODES; mi++) {
            uint32_t nid = edge->nodes[mi];
            if (nid >= net->n_nodes) continue;
            float a = scores[mi] / sum_exp;
            if (edge->attn_scores) edge->attn_scores[mi] = a;
            for (int d = 0; d < EMBED_DIM; d++)
                emsg[d] += a * net->nodes[nid].sig.vec[d];
        }
    }

    /* Phase 2: distribute edge messages back to active nodes */
    for (uint32_t i = 0; i < result->n_activated; i++) {
        uint32_t nid = result->activated_nodes[i];
        if (nid >= net->n_nodes || !net->nodes[nid].is_active) continue;

        float blended[EMBED_DIM] = {0};
        uint32_t n_contributing = 0;

        /* Gather from all fired hyperedges this node belongs to */
        for (uint32_t fi = 0; fi < result->n_fired; fi++) {
            uint32_t eid = result->fired_edges[fi];
            if (eid >= net->n_edges) continue;
            SRHNEdge4 *edge = &net->edges[eid];
            bool member = false;
            for (uint8_t mi = 0; mi < edge->n_nodes; mi++)
                if (edge->nodes[mi] == nid) { member = true; break; }
            if (!member) continue;

            float *emsg = mp->edge_msgs + (size_t)eid * EMBED_DIM;
            float act_w = srhn4_clampf(edge->activation, 0.f, 1.f);
            for (int d = 0; d < EMBED_DIM; d++) blended[d] += act_w * emsg[d];
            n_contributing++;
        }

        if (n_contributing == 0) continue;

        /* Normalise blended message */
        float mag = 0.f;
        for (int d = 0; d < EMBED_DIM; d++) mag += blended[d]*blended[d];
        mag = sqrtf(mag) + 1e-8f;
        for (int d = 0; d < EMBED_DIM; d++) blended[d] /= mag;

        /* Blend into node signature */
        SRHNNode4 *node = &net->nodes[nid];
        for (int d = 0; d < EMBED_DIM; d++)
            node->sig.vec[d] = (1.f - BLEND) * node->sig.vec[d] + BLEND * blended[d];

        /* Re-normalise node sig */
        mag = 0.f;
        for (int d = 0; d < EMBED_DIM; d++) mag += node->sig.vec[d]*node->sig.vec[d];
        mag = sqrtf(mag) + 1e-8f;
        for (int d = 0; d < EMBED_DIM; d++) node->sig.vec[d] /= mag;
        node->sig.magnitude = 1.f;

        /* Recompute orthogonal views from updated embedding */
        srhn4_orthogonalize_views(&node->sig);
    }

    /* Compute result-level message entropy */
    if (result->n_activated > 0) {
        float entropy = 0.f;
        float total_act = 0.f;
        for (uint32_t i = 0; i < result->n_activated; i++)
            total_act += result->activations[i];
        if (total_act > 1e-6f) {
            for (uint32_t i = 0; i < result->n_activated; i++) {
                float p = result->activations[i] / total_act;
                if (p > 1e-6f) entropy -= p * log2f(p);
            }
        }
        result->msg_entropy = entropy;
        net->stats.avg_msg_entropy = 0.9f * net->stats.avg_msg_entropy + 0.1f * entropy;
    }
}
