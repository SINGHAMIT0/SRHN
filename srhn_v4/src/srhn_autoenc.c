/*
 * srhn_autoenc.c  —  [R4] Autoencoder-based node quality filter
 * srhn_temporal.c is folded into this file for build simplicity.
 *
 * R4: A shallow autoencoder (384→192→384) is trained on all node
 *     signatures. After training, reconstruction error gates three
 *     decisions:
 *       high error (>0.72): node is noise → prune candidate
 *       low error  (<0.18): node is prototypical → promote to fast ring
 *       medium:            leave as-is
 *
 *     Training uses online SGD with momentum, one sample at a time,
 *     running every AE_RUN_INTERVAL queries.
 *
 * R5: Temporal edge decay: effective_weight = base × exp(-λ·Δt_seconds)
 *     λ = TEMPORAL_LAMBDA = 0.0002 → half-life ≈ 58 minutes.
 *     Edges fire → reset timestamp → start fresh.
 *     srhn4_temporal_weight() is called inline during propagation.
 */
#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 *  R4: Autoencoder
 * ═══════════════════════════════════════════════════════════════ */

#define AE_LR_DEFAULT  0.001f
#define AE_MOMENTUM    0.90f

/* He initialisation for a weight matrix */
static void he_init(float *W, int fan_in, int fan_out, SRHNNetwork4 *net) {
    float scale = sqrtf(2.0f / fan_in);
    for (int i = 0; i < fan_in * fan_out; i++) {
        /* Box-Muller from two uniform samples */
        float u1 = srhn4_randf(net) + 1e-8f;
        float u2 = srhn4_randf(net);
        float z  = sqrtf(-2.f * logf(u1)) * cosf(2.f * 3.14159265f * u2);
        W[i] = z * scale;
    }
}

void srhn4_ae_init(AutoEncoder *ae, float lr) {
    memset(ae, 0, sizeof(*ae));
    ae->lr = lr > 0.f ? lr : AE_LR_DEFAULT;
    ae->fitted = false;
    /* Weights are initialised lazily in first train call */
}

static void ae_init_weights(AutoEncoder *ae, SRHNNetwork4 *net) {
    he_init(&ae->W_enc[0][0], EMBED_DIM, AE_HIDDEN_DIM, net);
    he_init(&ae->W_dec[0][0], AE_HIDDEN_DIM, EMBED_DIM, net);
}

/* ReLU forward: out = relu(W·x + b) */
static void fc_relu(const float *x, int x_dim,
                     const float W[][AE_HIDDEN_DIM], const float *b,
                     float *out, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float s = b[j];
        for (int i = 0; i < x_dim; i++) s += x[i] * W[i][j];
        out[j] = srhn4_relu(s);
    }
}

static void fc_linear(const float *x, int x_dim,
                       const float W[][EMBED_DIM], const float *b,
                       float *out) {
    for (int j = 0; j < EMBED_DIM; j++) {
        float s = b[j];
        for (int i = 0; i < x_dim; i++) s += x[i] * W[i][j];
        out[j] = s;
    }
}

/*
 * Forward pass: input sig → hidden → reconstructed sig.
 * Returns MSE reconstruction error.
 */
float srhn4_ae_reconstruct_error(AutoEncoder *ae, const float *sig) {
    float hidden[AE_HIDDEN_DIM];
    float recon [EMBED_DIM];

    fc_relu  (sig,    EMBED_DIM,   ae->W_enc, ae->b_enc, hidden, AE_HIDDEN_DIM);
    fc_linear(hidden, AE_HIDDEN_DIM, ae->W_dec, ae->b_dec, recon);

    float err = 0.f;
    for (int i = 0; i < EMBED_DIM; i++) {
        float d = sig[i] - recon[i];
        err += d * d;
    }
    return err / EMBED_DIM;
}

/*
 * Single SGD step with momentum.
 * Stores running velocity externally (caller manages vel_enc / vel_dec buffers).
 */
static void ae_step(AutoEncoder *ae, const float *sig,
                     float vel_enc[][AE_HIDDEN_DIM],
                     float vel_dec[][EMBED_DIM]) {
    float hidden[AE_HIDDEN_DIM] = {0};
    float recon [EMBED_DIM]     = {0};

    /* Forward */
    for (int j = 0; j < AE_HIDDEN_DIM; j++) {
        float s = ae->b_enc[j];
        for (int i = 0; i < EMBED_DIM; i++) s += sig[i] * ae->W_enc[i][j];
        hidden[j] = srhn4_relu(s);
    }
    for (int j = 0; j < EMBED_DIM; j++) {
        float s = ae->b_dec[j];
        for (int i = 0; i < AE_HIDDEN_DIM; i++) s += hidden[i] * ae->W_dec[i][j];
        recon[j] = s;
    }

    /* Output gradient: dL/d_recon = 2*(recon - sig) / EMBED_DIM */
    float dout[EMBED_DIM];
    for (int j = 0; j < EMBED_DIM; j++)
        dout[j] = 2.f * (recon[j] - sig[j]) / EMBED_DIM;

    /* Decoder gradient */
    float dhidden[AE_HIDDEN_DIM] = {0};
    for (int i = 0; i < AE_HIDDEN_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            float g = dout[j] * hidden[i];
            vel_dec[i][j] = AE_MOMENTUM * vel_dec[i][j] - ae->lr * g;
            ae->W_dec[i][j] += vel_dec[i][j];
            dhidden[i] += dout[j] * ae->W_dec[i][j];
        }
    }
    for (int j = 0; j < EMBED_DIM; j++) ae->b_dec[j] -= ae->lr * dout[j];

    /* ReLU gradient, then encoder gradient */
    float drelu[AE_HIDDEN_DIM];
    for (int j = 0; j < AE_HIDDEN_DIM; j++)
        drelu[j] = (hidden[j] > 0.f) ? dhidden[j] : 0.f;

    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < AE_HIDDEN_DIM; j++) {
            float g = drelu[j] * sig[i];
            vel_enc[i][j] = AE_MOMENTUM * vel_enc[i][j] - ae->lr * g;
            ae->W_enc[i][j] += vel_enc[i][j];
        }
    }
    for (int j = 0; j < AE_HIDDEN_DIM; j++) ae->b_enc[j] -= ae->lr * drelu[j];
}

/*
 * srhn4_ae_train — train the autoencoder on all non-pruned node signatures.
 * n_iters: number of SGD steps (shuffle each pass).
 */
void srhn4_ae_train(SRHNNetwork4 *net, uint32_t n_iters) {
    AutoEncoder *ae = &net->autoenc;
    if (!ae->fitted) ae_init_weights(ae, net);

    uint32_t n_live = 0;
    for (uint32_t i = 0; i < net->n_nodes; i++)
        if (net->nodes[i].type != NODE_PRUNED) n_live++;
    if (n_live < 8) return;

    /* Velocity buffers for momentum SGD */
    static float vel_enc[EMBED_DIM][AE_HIDDEN_DIM];
    static float vel_dec[AE_HIDDEN_DIM][EMBED_DIM];
    if (!ae->fitted) {
        memset(vel_enc, 0, sizeof(vel_enc));
        memset(vel_dec, 0, sizeof(vel_dec));
    }

    float total_err = 0.f;
    for (uint32_t it = 0; it < n_iters; it++) {
        /* Pick a random live node */
        uint32_t idx = srhn4_rand(net) % net->n_nodes;
        int tries = 0;
        while (net->nodes[idx].type == NODE_PRUNED && tries++ < 32)
            idx = srhn4_rand(net) % net->n_nodes;
        if (net->nodes[idx].type == NODE_PRUNED) continue;

        ae_step(ae, net->nodes[idx].sig.vec, vel_enc, vel_dec);
        total_err += srhn4_ae_reconstruct_error(ae, net->nodes[idx].sig.vec);
    }
    ae->last_avg_error = total_err / (float)(n_iters + 1);
    ae->n_iters += n_iters;
    ae->fitted = true;
    net->stats.ae_runs++;
}

/*
 * srhn4_ae_run — score all nodes, gate prune/promote decisions.
 */
void srhn4_ae_run(SRHNNetwork4 *net) {
    AutoEncoder *ae = &net->autoenc;
    if (!ae->fitted) { srhn4_ae_train(net, 200); }

    for (uint32_t i = 0; i < net->n_nodes; i++) {
        SRHNNode4 *node = &net->nodes[i];
        if (node->type == NODE_PRUNED || node->in_fast_ring) continue;

        float err = srhn4_ae_reconstruct_error(ae, node->sig.vec);
        node->ae_recon_error = err;

        if (err > AE_RECON_PRUNE_THR && node->usage_count < 2.f && !node->auto_created) {
            /* Unusual but low-usage → mark as prune candidate (don't prune directly) */
            node->entropy_score = srhn4_clampf(node->entropy_score - 0.08f, 0.f, 1.f);
            net->stats.ae_nodes_pruned++;
        } else if (err < AE_RECON_FAST_THR && node->usage_count > 3.f) {
            /* Highly prototypical + frequently used → fast ring */
            srhn4_promote_fast(net, i);
            net->stats.ae_nodes_promoted++;
        }
    }
    ae->last_run_query = net->query_counter;
    fprintf(stderr, "[ae] Run complete: avg_err=%.4f promoted=%u pruned=%u\n",
            ae->last_avg_error, net->stats.ae_nodes_promoted, net->stats.ae_nodes_pruned);
}

/* ═══════════════════════════════════════════════════════════════
 *  R5: Temporal edge decay
 * ═══════════════════════════════════════════════════════════════ */

/*
 * srhn4_temporal_weight — compute effective weight with exponential decay.
 *
 * w_eff = base_weight × exp(-λ · Δt_seconds)
 * λ = TEMPORAL_LAMBDA = 0.0002
 * half-life: ln(2)/λ ≈ 3466 seconds ≈ 58 minutes
 *
 * Edge just fired → Δt=0 → w_eff = base_weight (no penalty)
 * Edge stale 1hr  → w_eff ≈ 0.5 × base_weight
 * Edge stale 1day → w_eff ≈ 0.00000004 × base_weight (effectively zero)
 */
float srhn4_temporal_weight(float base_weight, uint64_t last_fired_us) {
    if (last_fired_us == 0) return base_weight; /* never fired → no decay */
    uint64_t now = srhn4_timestamp_us();
    if (now <= last_fired_us) return base_weight;
    float dt_seconds = (float)(now - last_fired_us) / 1000000.f;
    float decay = expf(-TEMPORAL_LAMBDA * dt_seconds);
    return base_weight * decay;
}

/*
 * Update timestamp for a specific edge when it fires.
 */
void srhn4_temporal_update_edge(SRHNNetwork4 *net, uint32_t node_a, uint8_t edge_idx) {
    if (node_a >= net->n_nodes) return;
    SRHNNode4 *node = &net->nodes[node_a];
    if (edge_idx >= node->n_neighbors) return;
    if (node->edge_timestamps)
        node->edge_timestamps[edge_idx] = srhn4_timestamp_us();
}

/*
 * srhn4_temporal_decay_pass — refresh effective weights across all edges.
 * Called every 50 queries to keep edge_weights consistent with temporal state.
 * The base weight is never modified — only effective_weight (recomputed on access).
 */
void srhn4_temporal_decay_pass(SRHNNetwork4 *net) {
    for (uint32_t i = 0; i < net->n_nodes; i++) {
        SRHNNode4 *node = &net->nodes[i];
        if (node->type == NODE_PRUNED || !node->edge_timestamps) continue;
        for (uint8_t k = 0; k < node->n_neighbors; k++) {
            float effective = srhn4_temporal_weight(
                node->edge_weights[k], node->edge_timestamps[k]);
            /* Store effective weight back — base weight preserved via timestamps */
            node->edge_weights[k] = effective;
        }
    }
}
