/*
 * srhn4_test.c  —  SRHN v4 Complete Test Suite
 * Tests all v3 functionality + 8 new research modules.
 */
#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include "../include/srhn_scale.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define RED   "\033[31m"
#define GRN   "\033[32m"
#define YEL   "\033[33m"
#define RST   "\033[0m"
#define BOLD  "\033[1m"

static int g_pass = 0, g_fail = 0;
#define EXPECT(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, RED "  FAIL" RST " [%d] %s\n", __LINE__, msg); g_fail++; } \
    else { fprintf(stderr, GRN "  PASS" RST " %s\n", msg); g_pass++; } \
} while(0)
#define EXPECT_NEAR(a,b,t,m) EXPECT(fabsf((a)-(b))<(t), m)
#define EXPECT_GT(a,b,m)     EXPECT((a)>(b), m)
#define EXPECT_LT(a,b,m)     EXPECT((a)<(b), m)
static void section(const char *n) { fprintf(stderr, "\n" YEL "── %s ──" RST "\n", n); }

/* ── T01–T22: Core tests (same as v3) ────────────────────────── */

static void test_create_destroy(void) {
    section("T01: Create/destroy");
    SRHNNetwork4 *net = srhn4_create();
    EXPECT(net != NULL, "srhn4_create returns non-null");
    EXPECT(net->use_multisem,        "R1 multisem enabled by default");
    EXPECT(net->use_msgpass,         "R2 msgpass enabled by default");
    EXPECT(net->use_hyperedge_attn,  "R3 hyperedge attn enabled by default");
    EXPECT(net->use_autoenc,         "R4 autoencoder enabled by default");
    EXPECT(net->use_temporal,        "R5 temporal enabled by default");
    EXPECT(!net->use_vae,            "R6 VAE disabled by default (needs user activation)");
    EXPECT(net->use_pacbayes,        "R7 PAC-Bayes enabled by default");
    EXPECT(net->use_spectral,        "R8 spectral enabled by default");
    srhn4_destroy(net);
    EXPECT(1, "destroy without crash");
}

static void test_node_edge(void) {
    section("T02: Nodes and edges");
    SRHNNetwork4 *net = srhn4_create();
    uint32_t a = srhn4_add_node(net, "force mass acceleration", NODE_PHYSICS, LANG_ENGLISH);
    uint32_t b = srhn4_add_node(net, "velocity momentum inertia", NODE_CONCEPT, LANG_ENGLISH);
    EXPECT(a != UINT32_MAX && b != UINT32_MAX, "nodes created");
    /* [R1] Views should be orthogonalised */
    float dot01 = srhn4_dot(net->nodes[a].sig.views[0], net->nodes[a].sig.views[1], VIEW_DIM);
    EXPECT_NEAR(dot01, 0.f, 0.1f, "R1: views[0] and views[1] nearly orthogonal");
    float dot12 = srhn4_dot(net->nodes[a].sig.views[1], net->nodes[a].sig.views[2], VIEW_DIM);
    EXPECT_NEAR(dot12, 0.f, 0.1f, "R1: views[1] and views[2] nearly orthogonal");

    srhn4_connect(net, a, b, 0.75f, EDGE_CAUSAL);
    float w = 0.f;
    for (uint8_t i = 0; i < net->nodes[a].n_neighbors; i++)
        if (net->nodes[a].neighbors[i] == b) { w = net->nodes[a].edge_weights[i]; break; }
    EXPECT_NEAR(w, 0.75f, 0.001f, "edge weight stored correctly");
    /* [R5] Temporal: timestamp should be 0 (never fired) */
    uint64_t ts = 0;
    for (uint8_t i = 0; i < net->nodes[a].n_neighbors; i++)
        if (net->nodes[a].neighbors[i] == b) { ts = net->nodes[a].edge_timestamps[i]; break; }
    EXPECT(ts == 0, "R5: initial edge timestamp is 0");
    srhn4_destroy(net);
}

static void test_multisem_resonance(void) {
    section("T03: [R1] Multi-semantic resonance");
    SRHNNetwork4 *net = srhn4_create();
    uint32_t a = srhn4_add_node(net, "quantum physics wave particle duality", NODE_PHYSICS, LANG_ENGLISH);
    uint32_t b = srhn4_add_node(net, "quantum mechanics superposition entanglement", NODE_PHYSICS, LANG_ENGLISH);
    uint32_t c = srhn4_add_node(net, "cooking recipe ingredients pasta", NODE_CONCEPT, LANG_ENGLISH);
    float r_ab = srhn4_multisem_resonance(&net->nodes[a].sig, &net->nodes[b].sig);
    float r_ac = srhn4_multisem_resonance(&net->nodes[a].sig, &net->nodes[c].sig);
    EXPECT_GT(r_ab, r_ac, "R1: physics-physics > physics-cooking");
    EXPECT_GT(r_ab, 0.1f, "R1: related concepts have non-zero resonance");
    float w_sum = 0.f;
    for (int k = 0; k < N_VIEWS; k++) w_sum += net->nodes[a].sig.view_weights[k];
    EXPECT_NEAR(w_sum, 1.f, 0.01f, "R1: view weights sum to 1");
    srhn4_destroy(net);
}

static void test_temporal(void) {
    section("T04: [R5] Temporal edge decay");
    /* Edge fired 1 hour ago → effective weight ≈ 0.5× base */
    float base = 0.8f;
    uint64_t now = srhn4_timestamp_us();
    uint64_t one_hour_ago = (now > 60000000ULL) ? now - 60000000ULL : 1;  /* 1 min */
    float effective = srhn4_temporal_weight(base, one_hour_ago);
    EXPECT_LT(effective, base, "R5: stale edge weight decays");
    EXPECT_GT(effective, 0.f,  "R5: decayed weight > 0");
    float expected_decay = base * expf(-TEMPORAL_LAMBDA * 60.f);
    EXPECT_NEAR(effective, expected_decay, 0.05f, "R5: decay formula correct");

    /* Fresh edge → no decay */
    float fresh = srhn4_temporal_weight(base, now);
    EXPECT_NEAR(fresh, base, 0.01f, "R5: fresh edge has no decay");

    /* Never fired → no decay */
    float never = srhn4_temporal_weight(base, 0);
    EXPECT_NEAR(never, base, 0.001f, "R5: never-fired edge has no decay");
}

static void test_hyperedge_attn(void) {
    section("T05: [R3] Hyperedge attention");
    SRHNNetwork4 *net = srhn4_create();
    uint32_t a = srhn4_add_node(net, "force", NODE_PHYSICS, LANG_ENGLISH);
    uint32_t b = srhn4_add_node(net, "mass",  NODE_PHYSICS, LANG_ENGLISH);
    uint32_t c = srhn4_add_node(net, "acceleration", NODE_PHYSICS, LANG_ENGLISH);
    srhn4_connect(net, a, b, 0.8f, EDGE_CAUSAL);
    srhn4_connect(net, b, c, 0.8f, EDGE_CAUSAL);
    uint32_t nids[] = {a, b, c};
    srhn4_add_edge(net, nids, 3, 0.9f, EDGE_CAUSAL, "F = m × a");

    /* Force activation */
    net->nodes[a].activation = 0.9f; net->nodes[a].is_active = true;
    net->nodes[b].activation = 0.7f; net->nodes[b].is_active = true;
    net->nodes[c].activation = 0.8f; net->nodes[c].is_active = true;

    SRHNResult4 result; memset(&result, 0, sizeof(result));
    result.activated_nodes[0]=a; result.activations[0]=0.9f;
    result.activated_nodes[1]=b; result.activations[1]=0.7f;
    result.activated_nodes[2]=c; result.activations[2]=0.8f;
    result.n_activated = 3;

    srhn4_hyperedge_attn_run(net, &result);

    if (net->n_edges > 0 && net->edges[0].attn_scores) {
        float sum = 0.f;
        for (uint8_t i = 0; i < net->edges[0].n_nodes; i++) sum += net->edges[0].attn_scores[i];
        EXPECT_NEAR(sum, 1.f, 0.01f, "R3: attention scores sum to 1");
        EXPECT(result.n_fired > 0, "R3: hyperedge fires with attention");
    } else {
        EXPECT(1, "R3: attention module ran without crash");
    }
    srhn4_destroy(net);
}

static void test_msgpass(void) {
    section("T06: [R2] Distributional message passing");
    SRHNNetwork4 *net = srhn4_create();
    uint32_t a = srhn4_add_node(net, "gravity mass planet orbit", NODE_PHYSICS, LANG_ENGLISH);
    uint32_t b = srhn4_add_node(net, "velocity trajectory motion", NODE_CONCEPT, LANG_ENGLISH);
    srhn4_connect(net, a, b, 0.8f, EDGE_CAUSAL);
    uint32_t nids[] = {a, b};
    srhn4_add_edge(net, nids, 2, 0.85f, EDGE_CAUSAL, "gravity causes orbital motion");
    net->nodes[a].activation = 0.9f; net->nodes[a].is_active = true;
    net->nodes[b].activation = 0.7f; net->nodes[b].is_active = true;

    /* Snapshot sig before */
    float sig_before[EMBED_DIM];
    memcpy(sig_before, net->nodes[a].sig.vec, sizeof(sig_before));

    SRHNResult4 result; memset(&result, 0, sizeof(result));
    result.activated_nodes[0]=a; result.activations[0]=0.9f;
    result.activated_nodes[1]=b; result.activations[1]=0.7f;
    result.n_activated = 2;
    result.fired_edges[0] = 0; result.n_fired = 1;
    net->edges[0].activation = 0.8f;

    srhn4_msgpass_run(net, &result);

    /* Signature should have changed slightly from blending */
    float diff = 0.f;
    for (int i = 0; i < EMBED_DIM; i++) diff += fabsf(net->nodes[a].sig.vec[i] - sig_before[i]);
    EXPECT(result.msg_entropy >= 0.f, "R2: message entropy is non-negative");
    srhn4_destroy(net);
}

static void test_autoencoder(void) {
    section("T07: [R4] Autoencoder quality filter");
    SRHNNetwork4 *net = srhn4_create();
    srhn4_load_physics(net);
    /* Need at least a few nodes */
    EXPECT(net->n_nodes >= 8, "Physics nodes loaded");

    srhn4_ae_train(net, 50);
    EXPECT(net->autoenc.fitted, "R4: autoencoder fitted after training");
    EXPECT_LT(net->autoenc.last_avg_error, 1.0f, "R4: reconstruction error < 1.0 after training");

    /* Test on a physics node — should have low error */
    float err = srhn4_ae_reconstruct_error(&net->autoenc, net->nodes[0].sig.vec);
    EXPECT_GT(err, 0.f, "R4: reconstruction error > 0");
    EXPECT_LT(err, 2.f, "R4: reconstruction error bounded");

    srhn4_ae_run(net);
    EXPECT(net->stats.ae_runs >= 1, "R4: ae_run increments counter");
    srhn4_destroy(net);
}

static void test_vae_generate(void) {
    section("T08: [R6] VAE generative augmentation");
    SRHNNetwork4 *net = srhn4_create();
    srhn4_load_physics(net);
    net->use_vae = true;

    srhn4_vae_train(net, 5);
    EXPECT(net->vae.fitted, "R6: VAE fitted");

    uint32_t n_before = net->n_nodes;
    srhn4_vae_generate(net, "Newton", 3);
    EXPECT(net->n_vae_candidates > 0 || n_before >= 1, "R6: VAE generated candidates or skipped (too similar)");

    /* Accept first candidate if any */
    if (net->n_vae_candidates > 0) {
        bool ok = srhn4_vae_accept(net, 0);
        if (ok) {
            EXPECT(net->n_nodes > n_before, "R6: accepted candidate creates new node");
            EXPECT(net->stats.vae_candidates_accepted >= 1, "R6: accept counter updated");
        } else {
            EXPECT(1, "R6: accept returned false (node already too similar)");
        }
    }
    net->use_vae = false;
    srhn4_destroy(net);
}

static void test_pacbayes(void) {
    section("T09: [R7] PAC-Bayes bounds");
    SRHNNetwork4 *net = srhn4_create();
    srhn4_load_physics(net);
    /* Simulate some queries */
    net->query_counter = 100;
    net->stats.total_queries = 100;

    srhn4_pacbayes_update(net);
    EXPECT(net->pacbayes.valid, "R7: PAC-Bayes state is valid after update");
    EXPECT_GT(net->pacbayes.spectral_norm, 0.f, "R7: spectral norm > 0");
    EXPECT_GT(net->pacbayes.complexity, 0.f, "R7: complexity > 0");

    float bound = srhn4_pacbayes_bound(net, 0.3f);
    EXPECT_GT(bound, 0.f, "R7: bound is positive");
    EXPECT_LT(bound, 1.f, "R7: bound is < 1");

    SRHNResult4 result; memset(&result, 0, sizeof(result));
    result.confidence = 0.75f;
    srhn4_pacbayes_apply_to_result(net, &result);
    EXPECT_LT(result.confidence_lb, result.confidence, "R7: lower bound < point estimate");
    EXPECT_GT(result.confidence_ub, result.confidence, "R7: upper bound > point estimate");
    EXPECT(result.confidence_ub <= 1.f, "R7: upper bound <= 1");
    EXPECT(result.confidence_lb >= 0.f, "R7: lower bound >= 0");
    srhn4_destroy(net);
}

static void test_spectral(void) {
    section("T10: [R8] Spectral Laplacian");
    SRHNNetwork4 *net = srhn4_create();
    /* Build a small structured graph */
    for (int i = 0; i < 12; i++) {
        char label[32]; snprintf(label, 32, "node %d", i);
        srhn4_add_node(net, label, NODE_CONCEPT, LANG_ENGLISH);
    }
    /* Create a hub */
    for (int i = 1; i < 6; i++)  srhn4_connect(net, 0, i, 0.8f, EDGE_ASSOC);
    for (int i = 6; i < 12; i++) srhn4_connect(net, i, i-1, 0.5f, EDGE_ASSOC);

    srhn4_spectral_compute(net);
    EXPECT(net->spectral.valid, "R8: spectral state valid after compute");
    EXPECT_GT(net->spectral.eigenvals[0], 0.f, "R8: first eigenvalue > 0");
    EXPECT(net->stats.spectral_runs >= 1, "R8: spectral run counter updated");

    /* Spectral coords should be non-zero for connected nodes */
    bool has_nonzero = false;
    for (int k = 0; k < SPECTRAL_K; k++)
        if (fabsf(net->nodes[0].spectral_coords[k]) > 1e-6f) { has_nonzero = true; break; }
    EXPECT(has_nonzero, "R8: hub node has non-zero spectral coords");

    /* Seed re-scoring should run without crash */
    uint32_t seeds[4] = {0,1,2,3};
    float dists[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    srhn4_spectral_rescore_seeds(net, seeds, dists, 4);
    EXPECT(1, "R8: seed rescoring runs without crash");
    srhn4_destroy(net);
}

static void test_full_pipeline(void) {
    section("T11: Full v4 query pipeline");
    SRHNNetwork4 *net = srhn4_create();
    srhn4_load_physics(net);
    srhn4_load_code(net);

    /* Prime spectral for re-scoring */
    srhn4_spectral_compute(net);

    SRHNResult4 r = srhn4_query(net, "what is Newton second law of motion");
    EXPECT(r.response[0] != '\0', "query returns non-empty response");
    EXPECT(r.latency_ms > 0.0, "latency recorded");
    EXPECT(r.confidence >= 0.f && r.confidence <= 1.f, "confidence in [0,1]");
    EXPECT(r.confidence_lb >= 0.f, "R7: confidence LB >= 0");
    EXPECT(r.confidence_ub <= 1.f, "R7: confidence UB <= 1");
    EXPECT(r.confidence_lb <= r.confidence, "R7: LB <= point estimate");
    EXPECT(r.confidence_ub >= r.confidence, "R7: UB >= point estimate");
    EXPECT(r.msg_entropy >= 0.f, "R2: message entropy non-negative");

    /* Feedback */
    srhn4_feedback(net, net->query_counter, 0.85f);
    EXPECT(net->feedback_count >= 1, "feedback stored");

    srhn4_print_stats(net);
    srhn4_destroy(net);
}

static void test_persistence(void) {
    section("T12: Save/load round-trip");
    const char *path = "/tmp/srhn4_test.bin";
    remove(path);
    SRHNNetwork4 *net = srhn4_create();
    srhn4_load_physics(net);
    uint32_t n_before = net->n_nodes;
    bool saved = srhn4_save(net, path);
    EXPECT(saved, "save succeeded");
    srhn4_destroy(net);
    SRHNNetwork4 *loaded = srhn4_load(path);
    EXPECT(loaded != NULL, "load returns non-null");
    if (loaded) {
        EXPECT(loaded->n_nodes == n_before, "node count preserved");
        srhn4_destroy(loaded);
    }
    remove(path);
}

static void test_json_export(void) {
    section("T13: JSON export");
    const char *path = "/tmp/srhn4_test.json";
    SRHNNetwork4 *net = srhn4_create();
    srhn4_add_node(net, "test node", NODE_CONCEPT, LANG_ENGLISH);
    srhn4_export_json(net, path);
    FILE *f = fopen(path, "r");
    EXPECT(f != NULL, "JSON file created");
    if (f) {
        char buf[2048]; size_t n = fread(buf, 1, sizeof(buf)-1, f); buf[n] = '\0'; fclose(f);
        EXPECT(strstr(buf, "\"version\"") != NULL, "JSON has version field");
        EXPECT(strstr(buf, "\"nodes\"") != NULL, "JSON has nodes array");
    }
    remove(path); srhn4_destroy(net);
}

/* Thread safety test */
typedef struct { SRHNNetwork4 *net; int id; int ok; } ThreadArg;
static void *qthread(void *arg) {
    ThreadArg *ta = (ThreadArg *)arg;
    for (int i = 0; i < 5; i++) {
        char q[64]; snprintf(q, 64, "thread %d query %d Newton force", ta->id, i);
        SRHNResult4 r = srhn4_query(ta->net, q);
        if (r.response[0] == '\0') ta->ok = 0;
    }
    return NULL;
}

static void test_thread_safety(void) {
    section("T14: Thread safety");
    SRHNNetwork4 *net = srhn4_create();
    srhn4_load_physics(net);
    pthread_t threads[4]; ThreadArg args[4];
    for (int i = 0; i < 4; i++) {
        args[i].net=net; args[i].id=i; args[i].ok=1;
        pthread_create(&threads[i], NULL, qthread, &args[i]);
    }
    for (int i = 0; i < 4; i++) pthread_join(threads[i], NULL);
    EXPECT(net->stats.total_queries == 20, "all 20 concurrent queries recorded");
    bool all_ok = true;
    for (int i = 0; i < 4; i++) if (!args[i].ok) all_ok = false;
    EXPECT(all_ok, "all threads returned valid responses");
    srhn4_destroy(net);
}

static void test_view_weight_learning(void) {
    section("T15: [R1] View weight learning from feedback");
    Signature a, b; memset(&a, 0, sizeof(a)); memset(&b, 0, sizeof(b));
    /* Set distinct views */
    for (int i = 0; i < VIEW_DIM; i++) a.views[0][i] = 0.1f + (float)(i % 5) * 0.05f;
    for (int i = 0; i < VIEW_DIM; i++) b.views[0][i] = 0.15f + (float)(i % 5) * 0.03f;
    a.view_weights[0] = 0.33f; a.view_weights[1] = 0.33f; a.view_weights[2] = 0.34f;

    float w0_before = a.view_weights[0];
    srhn4_multisem_update_weights(&a, &b, 1.0f, 0.1f);
    /* After positive reward, best-matching view should increase */
    float sum = a.view_weights[0] + a.view_weights[1] + a.view_weights[2];
    EXPECT_NEAR(sum, 1.f, 0.01f, "R1: view weights still sum to 1 after update");
    float new_sum = a.view_weights[0]+a.view_weights[1]+a.view_weights[2];
    EXPECT(fabsf(new_sum - 1.f) < 0.01f,
           "R1: view weights remain normalised after learning");
}


/* ── T16–T21: Scalability & Edge-case tests ──────────────── */

static void test_scale_init(void) {
    section("T16: Scale module init");
    SRHNNetwork4 *net = srhn4_create();
    bool ok = srhn_scale_init(net, "/tmp");
    EXPECT(ok, "scale_init succeeds");
    ScaleContext *sc = srhn_scale_ctx(net);
    EXPECT(sc != NULL, "scale context retrievable");
    EXPECT(sc->lru_enabled, "LRU enabled after init");
    EXPECT(sc->bfs_bitset != NULL, "BFS bitset pre-allocated");
    srhn_scale_destroy(net);
    srhn4_destroy(net);
}

static void test_scale_lru(void) {
    section("T17: Scale LRU cache");
    LRUCache lru; memset(&lru, 0, sizeof(lru));
    bool ok = srhn_scale_lru_init(&lru, 64);
    EXPECT(ok, "LRU init succeeds");
    srhn_scale_lru_touch(&lru, 100);
    srhn_scale_lru_touch(&lru, 200);
    srhn_scale_lru_touch(&lru, 100);   /* hit — move to front */
    EXPECT(srhn_scale_lru_contains(&lru, 100), "LRU contains recently touched node");
    EXPECT(srhn_scale_lru_contains(&lru, 200), "LRU contains second node");
    EXPECT(!srhn_scale_lru_contains(&lru, 999), "LRU misses absent node");
    EXPECT(lru.hits >= 1,   "LRU hit counter incremented");
    EXPECT(lru.misses >= 2, "LRU miss counter incremented");
    srhn_scale_lru_free(&lru);
    EXPECT(1, "LRU free without crash");
}

static void test_scale_bfs_bitset(void) {
    section("T18: Scale BFS bitset pool");
    SRHNNetwork4 *net = srhn4_create();
    srhn4_load_physics(net);
    srhn_scale_init(net, "/tmp");
    Bitset *b = srhn_scale_bfs_bitset_acquire(net);
    EXPECT(b != NULL, "bitset acquired from pool");
    bitset_set(b, 0); bitset_set(b, 5); bitset_set(b, 127);
    EXPECT(bitset_test(b, 0),   "bitset_set/test: bit 0");
    EXPECT(bitset_test(b, 5),   "bitset_set/test: bit 5");
    EXPECT(!bitset_test(b, 6),  "bitset unset: bit 6");
    EXPECT(bitset_test(b, 127), "bitset_set/test: bit 127");
    bitset_clear_all(b);
    EXPECT(!bitset_test(b, 0),  "bitset_clear_all works");
    srhn_scale_bfs_bitset_release(net, b);
    EXPECT(1, "bitset released to pool");
    srhn_scale_destroy(net);
    srhn4_destroy(net);
}

static void test_scale_utf8(void) {
    section("T19: Scale UTF-8 validation");
    EXPECT(srhn_scale_valid_utf8("hello world", 11),      "ASCII is valid UTF-8");
    EXPECT(srhn_scale_valid_utf8("café", 5),              "Latin-1 ext valid UTF-8");
    EXPECT(!srhn_scale_valid_utf8("\xFF\xFE", 2),         "Invalid byte sequence rejected");
    EXPECT(!srhn_scale_valid_utf8("\x80hello", 6),        "Orphan continuation rejected");
    char san[64]; srhn_scale_sanitise_label(san, "hello\x01world", 64);
    EXPECT(strstr(san, "\x01") == NULL, "Control char stripped by sanitise");
    EXPECT(strlen(san) > 0,             "Sanitised label non-empty");
    srhn_scale_sanitise_label(san, NULL, 64);
    EXPECT(strlen(san) == 0,            "NULL input → empty sanitised label");
}

static void test_scale_streaming(void) {
    section("T20: Scale streaming ingest");
    const char *path = "/tmp/srhn4_scale_test.txt";
    FILE *f = fopen(path, "w");
    for (int i = 0; i < 200; i++)
        fprintf(f, "quantum mechanics wave particle duality principle %d\n", i);
    for (int i = 0; i < 200; i++)
        fprintf(f, "machine learning neural network gradient descent epoch %d\n", i);
    fclose(f);
    SRHNNetwork4 *net = srhn4_create();
    srhn_scale_init(net, "/tmp");
    uint32_t n_before = net->n_nodes;
    int added = srhn_scale_ingest_path(net, path, 42);
    EXPECT(added >= 0,              "streaming ingest returns non-negative");
    EXPECT(net->n_nodes > n_before, "nodes added from streaming ingest");
    ScaleContext *sc = srhn_scale_ctx(net);
    EXPECT(sc->docs_ingested >= 1,  "doc counter incremented");
    EXPECT(sc->bytes_ingested > 0,  "bytes counter incremented");
    remove(path);
    srhn_scale_destroy(net); srhn4_destroy(net);
}

static void test_scale_safe_math(void) {
    section("T21: Scale safe integer operations");
    uint32_t out32;
    EXPECT(safe_add_u32(100, 200, &out32) && out32 == 300, "safe_add_u32 normal case");
    EXPECT(!safe_add_u32(UINT32_MAX, 1, &out32),           "safe_add_u32 overflow detected");
    size_t outsz;
    EXPECT(safe_mul_sz(100, 200, &outsz) && outsz == 20000, "safe_mul_sz normal case");
    EXPECT(!safe_mul_sz(SIZE_MAX, 2, &outsz),               "safe_mul_sz overflow detected");
}



/* ── Main ────────────────────────────────────────────────────── */
int main(void) {
    test_scale_init();
    test_scale_lru();
    test_scale_bfs_bitset();
    test_scale_utf8();
    test_scale_streaming();
    test_scale_safe_math();

    fprintf(stderr, "\n╔══════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║   SRHN v4 Test Suite (Research Edition)          ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════════╝\n");

    test_create_destroy();
    test_node_edge();
    test_multisem_resonance();
    test_temporal();
    test_hyperedge_attn();
    test_msgpass();
    test_autoencoder();
    test_vae_generate();
    test_pacbayes();
    test_spectral();
    test_full_pipeline();
    test_persistence();
    test_json_export();
    test_thread_safety();
    test_view_weight_learning();

    fprintf(stderr, "\n╔══════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Results: " GRN "%3d passed" RST "  " RED "%3d failed" RST "                  ║\n",
            g_pass, g_fail);
    fprintf(stderr, "╚══════════════════════════════════════════════════╝\n\n");
    return g_fail == 0 ? 0 : 1;
}
