/*
 * srhn_memory.c  —  Memory systems for SRHN v3
 *
 * Implements:
 *   1. Working memory — 16-slot high-salience buffer with activation bonus
 *   2. Episodic memory — unbounded session history with similarity recall
 *   3. Context blending — cross-turn signature accumulation
 */

#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════
 *  Working Memory
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Insert a node into working memory.
 * If all slots are full, evict the node with lowest salience.
 * Nodes in working memory get 10x activation bonus during propagation.
 */
void srhn4_working_mem_insert(SRHNNetwork4 *net, uint32_t node_id, float salience) {
    WorkingMemory *wm = &net->working_mem;
    salience = srhn4_clampf(salience, 0.f, 1.f);

    /* Check if already present — update salience */
    for (uint8_t i = 0; i < wm->n_slots; i++) {
        if (wm->slot_nodes[i] == node_id) {
            wm->slot_salience[i] = srhn4_clampf(wm->slot_salience[i] + salience, 0.f, 1.f);
            return;
        }
    }

    if (wm->n_slots < WORKING_MEM_SLOTS) {
        /* Add to empty slot */
        wm->slot_nodes   [wm->n_slots] = node_id;
        wm->slot_salience[wm->n_slots] = salience;
        wm->n_slots++;
        return;
    }

    /* Evict lowest-salience slot */
    float min_s = wm->slot_salience[0];
    uint8_t min_i = 0;
    for (uint8_t i = 1; i < wm->n_slots; i++) {
        if (wm->slot_salience[i] < min_s) {
            min_s = wm->slot_salience[i]; min_i = i;
        }
    }

    if (salience > min_s) {
        wm->slot_nodes   [min_i] = node_id;
        wm->slot_salience[min_i] = salience;
    }
}

/* Clear working memory (e.g., at session boundary) */
void srhn4_working_mem_clear(SRHNNetwork4 *net) {
    WorkingMemory *wm = &net->working_mem;
    wm->n_slots = 0;
    memset(wm->slot_nodes,    0, sizeof(wm->slot_nodes));
    memset(wm->slot_salience, 0, sizeof(wm->slot_salience));
    wm->last_cleared_us = srhn4_timestamp_us();
}

/*
 * Apply working memory — decay salience each query.
 * This simulates forgetting; nodes not reinforced fade from working memory.
 */
void srhn4_working_mem_apply(SRHNNetwork4 *net) {
    WorkingMemory *wm = &net->working_mem;
    uint8_t new_n = 0;

    for (uint8_t i = 0; i < wm->n_slots; i++) {
        wm->slot_salience[i] *= 0.80f; /* 20% decay per query */
        if (wm->slot_salience[i] > 0.05f) {
            /* Keep */
            wm->slot_nodes   [new_n] = wm->slot_nodes   [i];
            wm->slot_salience[new_n] = wm->slot_salience[i];
            new_n++;
        }
    }
    wm->n_slots = new_n;
}

/* ═══════════════════════════════════════════════════════════════
 *  Episodic Memory
 * ═══════════════════════════════════════════════════════════════ */

bool srhn4_episodic_init(SRHNNetwork4 *net, uint32_t cap) {
    EpisodicMemory *em = &net->episodic;
    em->cap      = cap > 0 ? cap : EPISODIC_CAP;
    em->episodes = (Episode *)calloc(em->cap, sizeof(Episode));
    em->n_episodes = 0;
    em->head     = 0;
    if (!em->episodes) return false;
    return true;
}

void srhn4_episodic_free(SRHNNetwork4 *net) {
    EpisodicMemory *em = &net->episodic;
    free(em->episodes);
    em->episodes   = NULL;
    em->n_episodes = 0;
    em->cap        = 0;
    em->head       = 0;
}

/*
 * Store a query+response episode.
 * Uses circular buffer — overwrites oldest on overflow.
 */
void srhn4_episodic_store(SRHNNetwork4 *net, SRHNResult4 *result,
                           const char *query, float reward) {
    EpisodicMemory *em = &net->episodic;
    if (!em->episodes) return;

    Episode *ep = &em->episodes[em->head % em->cap];
    memset(ep, 0, sizeof(Episode));

    ep->timestamp_us = srhn4_timestamp_us();
    strncpy(ep->query,    query,           511);
    strncpy(ep->response, result->response,1023);
    ep->confidence = result->confidence;
    ep->lang       = result->response_lang;
    ep->reward     = reward;

    /* Store top key nodes */
    ep->n_key = 0;
    for (uint32_t i = 0; i < result->n_activated && ep->n_key < 16; i++) {
        if (result->activations[i] > 0.3f)
            ep->key_nodes[ep->n_key++] = result->activated_nodes[i];
    }

    em->head++;
    if (em->n_episodes < em->cap) em->n_episodes++;
}

/*
 * Recall K most semantically similar episodes to the current query.
 * Uses cosine similarity over a compact episode signature (sum of key node sigs).
 * Returns pointer array into em->episodes — do not free.
 */
void srhn4_episodic_recall(SRHNNetwork4 *net, const Signature *query_sig,
                            int k, Episode **out, int *n_out) {
    EpisodicMemory *em = &net->episodic;
    *n_out = 0;
    if (!em->episodes || em->n_episodes == 0 || !out) return;

    typedef struct { Episode *ep; float sim; } EpSim;
    EpSim *scores = (EpSim *)malloc(em->n_episodes * sizeof(EpSim));
    if (!scores) return;
    int n_sc = 0;

    uint32_t ep_count = em->n_episodes < em->cap ? em->n_episodes : em->cap;
    for (uint32_t i = 0; i < ep_count; i++) {
        Episode *ep = &em->episodes[i];
        if (!ep->query[0]) continue;

        /* Compute episode signature: mean of key node signatures */
        float ep_vec[EMBED_DIM];
        memset(ep_vec, 0, sizeof(ep_vec));
        int valid = 0;
        for (uint8_t j = 0; j < ep->n_key; j++) {
            uint32_t nid = ep->key_nodes[j];
            if (nid >= net->n_nodes || net->nodes[nid].type == NODE_PRUNED) continue;
            for (int d = 0; d < EMBED_DIM; d++)
                ep_vec[d] += net->nodes[nid].sig.vec[d];
            valid++;
        }

        float sim = 0.f;
        if (valid > 0) {
            float mag = 0.f;
            for (int d = 0; d < EMBED_DIM; d++) {
                ep_vec[d] /= valid;
                mag += ep_vec[d] * ep_vec[d];
            }
            mag = sqrtf(mag) + 1e-8f;
            for (int d = 0; d < EMBED_DIM; d++) ep_vec[d] /= mag;
            sim = srhn4_dot(query_sig->vec, ep_vec, EMBED_DIM);
            if (sim < 0.f) sim = 0.f;
        } else {
            /* Fallback: text similarity via query string hash */
            uint32_t h = 5381;
            for (const char *p = ep->query; *p; p++)
                h = ((h << 5) + h) ^ (uint8_t)*p;
            uint32_t qh = 5381;
            /* Can't access query string here, use stored sig hash */
            sim = 1.f - (float)__builtin_popcount(query_sig->hash ^ h) / 32.f;
            if (sim < 0.f) sim = 0.f;
        }

        /* Recency bonus: more recent episodes are slightly preferred */
        uint64_t age_us = srhn4_timestamp_us() - ep->timestamp_us;
        float recency = expf(-(float)(age_us / 1000000ULL) / 3600.f); /* decay over hours */
        sim = sim * 0.85f + recency * 0.15f;

        scores[n_sc++] = (EpSim){ep, sim};
    }

    /* Partial sort: top-k by similarity */
    for (int i = 0; i < k && i < n_sc; i++) {
        int max_j = i;
        for (int j = i + 1; j < n_sc; j++)
            if (scores[j].sim > scores[max_j].sim) max_j = j;
        EpSim tmp = scores[i]; scores[i] = scores[max_j]; scores[max_j] = tmp;
        out[(*n_out)++] = scores[i].ep;
    }

    free(scores);
}

/*
 * Pre-activate nodes from top recalled episodes.
 * Called at query start to warm up the graph with prior context.
 */
static void apply_episodic_warmup(SRHNNetwork4 *net, const Signature *query_sig) {
    Episode *recalled[4];
    int n_recalled = 0;
    srhn4_episodic_recall(net, query_sig, 4, recalled, &n_recalled);

    for (int e = 0; e < n_recalled; e++) {
        Episode *ep = recalled[e];
        float reward_bonus = srhn4_clampf(ep->reward * 0.3f + 0.1f, 0.f, 0.3f);
        for (uint8_t k = 0; k < ep->n_key; k++) {
            uint32_t nid = ep->key_nodes[k];
            if (nid >= net->n_nodes || net->nodes[nid].type == NODE_PRUNED) continue;
            /* Slight pre-activation from episodic recall */
            net->nodes[nid].activation = srhn4_clampf(
                net->nodes[nid].activation + reward_bonus, 0.f, 0.4f);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  Context signature blending
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Build an "effective query signature" that blends the current query
 * with recent conversation history using exponential decay weights.
 *
 * effective_sig ≈ 0.5*q + 0.25*q_{t-1} + 0.125*q_{t-2} + 0.0625*q_{t-3}
 *
 * This enables multi-turn coherence: "what's its formula?" works after
 * "tell me about Newton's second law" because the session context still
 * contains the Newton signature.
 */
void srhn4_ctx_blend_query(SRHNNetwork4 *net, Signature *effective_sig,
                            const Signature *query_sig) {
    /* Start with current query */
    *effective_sig = *query_sig;

    if (net->context.count == 0) return;

    /* Blend with up to 4 prior turns */
    float alpha = 0.25f;
    for (int t = 0; t < 4 && t < net->context.count; t++) {
        int idx = (net->context.head - 1 - t + CTX_HISTORY) % CTX_HISTORY;
        CtxTurn4 *turn = &net->context.turns[idx];

        if (!turn->query[0]) { alpha *= 0.5f; continue; }

        srhn4_sig_blend(effective_sig, &turn->blended_sig, alpha);
        alpha *= 0.5f;
    }

    /* Also blend with session-level signature */
    srhn4_sig_blend(effective_sig, &net->context.session_sig, 0.08f);

    /* Apply episodic warmup */
    apply_episodic_warmup(net, effective_sig);
}

/*
 * Update context after a query completes.
 * Stores turn, updates session signature.
 */
void srhn4_ctx_update(SRHNNetwork4 *net, const char *query, SRHNResult4 *result,
                       const Signature *query_sig) {
    ConvContext *ctx = &net->context;
    CtxTurn4 *turn   = &ctx->turns[ctx->head % CTX_HISTORY];

    strncpy(turn->query,    query,           511);
    strncpy(turn->response, result->response, RESP_BUF_SIZE - 1);
    turn->confidence  = result->confidence;
    turn->lang        = result->response_lang;
    turn->timestamp_us= srhn4_timestamp_us();
    turn->n_activated = result->n_activated < CTX_TOKEN_CAP
                         ? result->n_activated : CTX_TOKEN_CAP;
    memcpy(turn->activated, result->activated_nodes,
           turn->n_activated * sizeof(uint32_t));
    turn->blended_sig = *query_sig;

    ctx->head  = (ctx->head + 1) % CTX_HISTORY;
    if (ctx->count < CTX_HISTORY) ctx->count++;
    ctx->session_lang = result->response_lang;

    /* Update rolling session signature */
    srhn4_sig_blend(&ctx->session_sig, query_sig, 0.15f);

    /* Insert top activated nodes into working memory */
    for (uint32_t i = 0; i < result->n_activated && i < 4; i++) {
        float sal = result->activations[i];
        if (sal > 0.25f)
            srhn4_working_mem_insert(net, result->activated_nodes[i], sal);
    }
}
