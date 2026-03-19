/*
 * srhn_v4.h  —  SRHN v4: Research-Enhanced Semantic Reasoning Hypergraph
 *
 * New in v4 (8 research improvements from MSH-FSL + HGNN survey 2025):
 *
 *  [R1] Multi-semantic orthogonal views   (srhn_multisem.c)
 *       Inspired by MSH-FSL orthogonalized mapping. Each node stores
 *       N_VIEWS independent 384-dim sub-signatures via Gram-Schmidt.
 *
 *  [R2] Distributional message passing    (srhn_msgpass.c)
 *       Replaces scalar BFS spike-decay with per-edge attention-weighted
 *       signature distribution blending (bidirectional node↔hyperedge).
 *
 *  [R3] Hyperedge attention               (srhn_hyperedge_attn.c)
 *       Softmax-attention scoring per member node inside each hyperedge,
 *       replacing the binary "all-or-nothing" causal fire rule.
 *
 *  [R4] Autoencoder node quality filter   (srhn_autoenc.c)
 *       Periodic encoder-decoder over node signatures; reconstruction
 *       error gates prune vs. fast-ring promotion decisions.
 *
 *  [R5] Temporal edge decay               (srhn_temporal.c)
 *       Edge effective weight = stored_weight × exp(-λ·Δt). Stale
 *       connections fade; recent paths strengthen automatically.
 *
 *  [R6] Lightweight hypergraph VAE        (srhn_vae.c)
 *       Generative augmentation for sparse domains: encodes node
 *       clusters, samples nearby latent vectors, decodes into candidate
 *       nodes for human-in-the-loop insertion.
 *
 *  [R7] PAC-Bayes confidence bounds       (srhn_pacbayes.c)
 *       Derives theoretical confidence bounds from graph spectral norms
 *       and hyperedge density. Reported alongside calibrated confidence.
 *
 *  [R8] Spectral Laplacian seed scoring   (srhn_spectral.c)
 *       Combines HNSW metric seeds with hypergraph Laplacian eigenvectors
 *       for structure-aware seed re-ranking.
 */

#pragma once
#ifndef SRHN_V4_H
#define SRHN_V4_H

#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

/* ═══════════════════════════════════════════════════════
 *  Version
 * ═══════════════════════════════════════════════════════ */
#define SRHN4_VERSION       "4.0.0"
#define SRHN4_MAGIC         0x53524834u   /* "SRH4" */

/* ═══════════════════════════════════════════════════════
 *  Core dimensions & limits
 * ═══════════════════════════════════════════════════════ */
#define EMBED_DIM           384
#define EMBED_DIM_GLOVE     300
#define MAX_WORD_LEN        64
#define MAX_TOKEN_LEN       32
#define MAX_VOCAB           65536

/* [R1] multi-semantic views per node */
#define N_VIEWS             3             /* orthogonal semantic sub-spaces */
#define VIEW_DIM            128           /* each view = 128-dim (3×128=384) */

/* Graph limits */
#define SRHN4_INIT_NODES    2048
#define SRHN4_MAX_NODES     1048576
#define SRHN4_INIT_EDGES    4096
#define SRHN4_MAX_EDGES     4194304
#define MAX_NEIGHBORS       256
#define FAST_RING_CAP       2048
#define MAX_EDGE_NODES      16

/* HNSW */
#define HNSW_M              16
#define HNSW_M0             32
#define HNSW_EF_CONSTRUCTION 200
#define HNSW_EF_SEARCH      64
#define HNSW_MAX_LAYERS     16

/* Reasoning */
#define MAX_HOPS            12
#define MAX_CHAIN_LEN       16
#define MAX_CHAINS          8
#define MAX_ACTIVATED       1024
#define RESONANCE_THRESH    0.12f
#define SPIKE_DECAY         0.72f
#define PRUNE_THRESH        0.08f

/* Memory */
#define CTX_HISTORY         64
#define CTX_TOKEN_CAP       128
#define WORKING_MEM_SLOTS   16
#define EPISODIC_CAP        10000
#define FEEDBACK_RING       512
#define WAL_FLUSH_INTERVAL  50

/* LLM */
#define LLM_CTX_SIZE        4096
#define LLM_MAX_TOKENS      2048
#define PROMPT_TEMPLATE_CAP 8

/* API */
#define API_DEFAULT_PORT    8765
#define API_MAX_CONN        64
#define API_READ_BUF        65536

/* Response buffers */
#define RESP_BUF_SIZE       16384
#define CONTEXT_BUF_SIZE    8192

/* [R4] Autoencoder */
#define AE_HIDDEN_DIM       192
#define AE_RUN_INTERVAL     200    /* queries between autoencoder passes */
#define AE_RECON_PRUNE_THR  0.72f  /* reconstruction error → prune candidate */
#define AE_RECON_FAST_THR   0.18f  /* very low error → promote to fast ring  */

/* [R5] Temporal decay */
#define TEMPORAL_LAMBDA     0.0002f  /* decay per second: w_eff = w × e^(-λ·Δt) */

/* [R6] VAE */
#define VAE_LATENT_DIM      64
#define VAE_HIDDEN_DIM      256
#define VAE_MIN_DOMAIN_NODES 8  /* generate only when domain has < this many nodes */

/* [R7] PAC-Bayes */
#define PACBAYES_DELTA      0.05f  /* confidence level (95%) */

/* [R8] Spectral */
#define SPECTRAL_K          8      /* number of Laplacian eigenvectors to use */
#define SPECTRAL_RECOMPUTE  500    /* queries between spectral recomputation */

/* Calibration */
#define CALIB_BUCKETS       20
#define PAGERANK_INTERVAL   100

/* ═══════════════════════════════════════════════════════
 *  Enumerations (unchanged from v3)
 * ═══════════════════════════════════════════════════════ */
typedef enum {
    LANG_UNKNOWN=0, LANG_ENGLISH, LANG_HINDI, LANG_SPANISH,
    LANG_FRENCH, LANG_GERMAN, LANG_ARABIC, LANG_CHINESE,
    LANG_JAPANESE, LANG_RUSSIAN, LANG_PORTUGUESE, LANG_ITALIAN,
    LANG_BENGALI, LANG_CODE, LANG_MIXED, LANG_COUNT
} LangID;

typedef enum {
    NODE_CONCEPT=0, NODE_ENTITY, NODE_EVENT, NODE_RELATION,
    NODE_PHYSICS, NODE_CODE, NODE_VISION, NODE_EMOTION,
    NODE_PROCEDURE, NODE_PRUNED
} NodeType;

typedef enum {
    EDGE_ASSOC=0, EDGE_CAUSAL, EDGE_SYNONYM, EDGE_ANTONYM,
    EDGE_INSTANCE, EDGE_PART_OF, EDGE_CAUSES, EDGE_INHIBITORY,
    EDGE_TEMPORAL, EDGE_SPATIAL, EDGE_CODE, EDGE_VISUAL,
    EDGE_CONTRADICTS, EDGE_SUPPORTS, EDGE_UNKNOWN
} EdgeType;

typedef enum {
    POS_UNKNOWN=0, POS_NOUN, POS_VERB, POS_ADJ,
    POS_ADV, POS_PREP, POS_CONJ, POS_PRON
} PartOfSpeech;

typedef enum {
    SRHN_OK=0, SRHN_ERR_NULL_PTR, SRHN_ERR_OOM, SRHN_ERR_BOUNDS,
    SRHN_ERR_DUPLICATE, SRHN_ERR_IO, SRHN_ERR_CORRUPT, SRHN_ERR_LLM,
    SRHN_ERR_EMBED, SRHN_ERR_FULL, SRHN_ERR_NOT_FOUND, SRHN_ERR_LOCK
} SRHNError;

/* ═══════════════════════════════════════════════════════
 *  [R1] Multi-Semantic Signature
 *  Each node carries N_VIEWS orthogonal sub-signatures.
 *  views[0] = primary (physics/causal facet)
 *  views[1] = secondary (linguistic/contextual facet)
 *  views[2] = tertiary (domain-specific / emotional facet)
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    float    vec[EMBED_DIM];   /* primary full-dim vector (backward compat) */
    float    views[N_VIEWS][VIEW_DIM]; /* orthogonal sub-vectors */
    float    view_weights[N_VIEWS];    /* learned importance per view */
    float    magnitude;
    uint32_t hash;
    LangID   lang;
} Signature;

/* ═══════════════════════════════════════════════════════
 *  [R5] Temporal edge info
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    uint64_t last_fired_us;    /* monotonic timestamp of last fire */
    float    base_weight;      /* stored weight (before temporal decay) */
    float    effective_weight; /* = base × exp(-λ·Δt), recomputed on use */
    uint32_t fire_count;
} TemporalEdge;

/* ═══════════════════════════════════════════════════════
 *  Node (extended for v4)
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    uint32_t  id;
    NodeType  type;
    LangID    lang;
    char      label[128];

    /* [R1] Multi-semantic signature */
    Signature sig;

    /* Activation */
    float     activation;
    float     activation_peak;
    bool      is_active;
    float     usage_count;
    float     last_active;

    /* Graph */
    uint32_t *neighbors;
    float    *edge_weights;       /* base weights (before temporal decay) */
    EdgeType *edge_types;
    uint64_t *edge_timestamps;    /* [R5] last-fired time per edge */
    uint8_t   n_neighbors;
    uint16_t  neighbor_cap;

    /* Quality */
    float     entropy_score;
    float     importance;         /* PageRank */
    float     ae_recon_error;     /* [R4] last autoencoder reconstruction error */
    bool      in_fast_ring;
    bool      auto_created;
    bool      ae_candidate;       /* [R4] flagged for human review (VAE-generated) */
    uint32_t  source_id;
    uint32_t  proto_id;
    uint32_t  lex_id;

    /* Domain-specific */
    char      formula[128];
    char      units[32];
    char      code_lang[16];
    char      code_pattern[256];

    /* [R8] Spectral */
    float     spectral_coords[SPECTRAL_K]; /* projection onto Laplacian eigenvectors */

    /* HNSW */
    int8_t    hnsw_layer;
    float     sentiment;
} SRHNNode4;

/* ═══════════════════════════════════════════════════════
 *  Hyperedge (extended for v4)
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    uint32_t  id;
    uint32_t *nodes;
    uint8_t   n_nodes;
    float     weight;
    float     causal_strength;
    float     activation;

    /* [R3] Hyperedge attention */
    float    *attn_scores;        /* softmax score per member node */
    float     attn_entropy;       /* entropy of attention distribution */

    EdgeType  type;
    bool      is_causal;
    uint32_t  fire_count;
    char      relation[128];
    uint64_t  last_fired_us;      /* [R5] temporal tracking */
} SRHNEdge4;

/* ═══════════════════════════════════════════════════════
 *  Lexicon
 * ═══════════════════════════════════════════════════════ */
#define MAX_SYNONYMS 8
#define MAX_ABBREV   4

typedef struct {
    char        word[MAX_WORD_LEN];
    LangID      lang;
    PartOfSpeech pos;
    float       sentiment, frequency;
    char        synonyms [MAX_SYNONYMS][MAX_WORD_LEN];
    char        antonyms [4][MAX_WORD_LEN];
    char        abbrevs  [MAX_ABBREV][MAX_WORD_LEN];
    uint8_t     n_synonyms, n_antonyms, n_abbrevs;
} LexEntry;

typedef struct {
    LexEntry *entries;
    uint32_t  n_entries, cap;
} Lexicon;

typedef struct {
    char  name[64], formula[128], domain[32], units[64], description[256];
    float importance;
} PhysicsLaw;

/* ═══════════════════════════════════════════════════════
 *  Reasoning
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    uint32_t nodes    [MAX_CHAIN_LEN];
    char     labels   [MAX_CHAIN_LEN][128];
    char     relations[MAX_CHAIN_LEN][64];
    EdgeType edge_types[MAX_CHAIN_LEN];
    float    strengths[MAX_CHAIN_LEN];
    float    total_strength;
    uint8_t  length;
    bool     is_causal;
    bool     has_contradiction;
} ReasoningChain;

typedef struct {
    uint32_t node_a, node_b;
    float    confidence;
    char     label_a[128], label_b[128];
} Contradiction;

/* ═══════════════════════════════════════════════════════
 *  Query Result (extended for v4)
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    uint32_t activated_nodes[MAX_ACTIVATED];
    float    activations    [MAX_ACTIVATED];
    uint32_t n_activated;

    uint32_t fired_edges[256];
    uint32_t n_fired;

    ReasoningChain chains[MAX_CHAINS];
    uint8_t        n_chains;

    Contradiction  contradictions[4];
    uint8_t        n_contradictions;

    char     top_concepts[8][128];
    uint8_t  n_top_concepts;

    char     llm_context[CONTEXT_BUF_SIZE];
    char     response   [RESP_BUF_SIZE];
    LangID   response_lang;
    bool     response_grounded;

    /* Confidence (v4: two signals) */
    float    confidence;           /* isotonic-calibrated */
    float    confidence_lb;        /* [R7] PAC-Bayes lower bound */
    float    confidence_ub;        /* [R7] PAC-Bayes upper bound */

    /* [R2] Distributional message passing info */
    float    msg_entropy;          /* Shannon entropy of final activation dist */

    /* [R3] Hyperedge attention info */
    float    max_attn_score;       /* strongest single node-in-hyperedge attention */

    /* Metrics */
    float    compute_used;
    double   latency_ms, llm_latency_ms;
    uint32_t hops_taken, seeds_used;
    float    best_resonance;
    bool     used_llm;
} SRHNResult4;

/* ═══════════════════════════════════════════════════════
 *  [R2] Distributional Message Passing state
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    float    *node_msgs;        /* n_nodes × EMBED_DIM messages buffer */
    float    *edge_msgs;        /* n_edges × EMBED_DIM messages buffer */
    float    *attn_weights;     /* per (node,edge) attention weight */
    uint32_t  n_nodes_alloc;
    uint32_t  n_edges_alloc;
    float     temperature;      /* softmax temperature (default 1.0) */
    bool      enabled;
} MsgPassState;

/* ═══════════════════════════════════════════════════════
 *  [R4] Autoencoder state
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    /* Encoder: EMBED_DIM → AE_HIDDEN_DIM */
    float W_enc [EMBED_DIM][AE_HIDDEN_DIM];
    float b_enc [AE_HIDDEN_DIM];
    /* Decoder: AE_HIDDEN_DIM → EMBED_DIM */
    float W_dec [AE_HIDDEN_DIM][EMBED_DIM];
    float b_dec [EMBED_DIM];
    /* Training state */
    float lr;
    uint32_t n_iters;
    float last_avg_error;
    uint32_t last_run_query;
    bool fitted;
} AutoEncoder;

/* ═══════════════════════════════════════════════════════
 *  [R6] VAE state
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    /* Encoder: EMBED_DIM → 2×VAE_LATENT_DIM (mean + logvar) */
    float W_enc [EMBED_DIM][VAE_HIDDEN_DIM];
    float b_enc [VAE_HIDDEN_DIM];
    float W_mu  [VAE_HIDDEN_DIM][VAE_LATENT_DIM];
    float b_mu  [VAE_LATENT_DIM];
    float W_lv  [VAE_HIDDEN_DIM][VAE_LATENT_DIM];
    float b_lv  [VAE_LATENT_DIM];
    /* Decoder: VAE_LATENT_DIM → VAE_HIDDEN_DIM → EMBED_DIM */
    float W_dec1[VAE_LATENT_DIM][VAE_HIDDEN_DIM];
    float b_dec1[VAE_HIDDEN_DIM];
    float W_dec2[VAE_HIDDEN_DIM][EMBED_DIM];
    float b_dec2[EMBED_DIM];
    /* State */
    float lr;
    uint32_t n_epochs;
    float last_elbo;
    bool fitted;
} HypergraphVAE;

/* Candidate nodes generated by VAE for human review */
typedef struct {
    float     sig_vec[EMBED_DIM];
    char      label_suggestion[128];
    float     novelty_score;       /* distance from nearest existing node */
    NodeType  suggested_type;
    LangID    lang;
    bool      accepted;
} VAECandidate;

/* ═══════════════════════════════════════════════════════
 *  [R7] PAC-Bayes state
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    float spectral_norm;       /* largest singular value of weight matrix */
    float hyperedge_density;   /* avg hyperedge degree / n_nodes */
    float complexity;          /* PAC-Bayes complexity term */
    float delta;               /* confidence parameter (0.05 = 95%) */
    uint32_t n_samples_seen;   /* denominator for bound tightness */
    bool  valid;               /* has been computed at least once */
} PACBayesState;

/* ═══════════════════════════════════════════════════════
 *  [R8] Spectral Laplacian state
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    float    *eigenvecs;       /* SPECTRAL_K × n_nodes eigenvector matrix */
    float    *eigenvals;       /* SPECTRAL_K eigenvalues */
    uint32_t  n_nodes_alloc;
    uint32_t  last_run_query;
    bool      valid;
    pthread_mutex_t lock;
} SpectralState;

/* ═══════════════════════════════════════════════════════
 *  Conversation & Memory (unchanged from v3)
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    char     query   [512];
    char     response[RESP_BUF_SIZE];
    float    confidence;
    LangID   lang;
    uint64_t timestamp_us;
    uint32_t activated[CTX_TOKEN_CAP];
    uint32_t n_activated;
    Signature blended_sig;
} CtxTurn4;

typedef struct {
    CtxTurn4  turns[CTX_HISTORY];
    uint8_t   head, count;
    LangID    session_lang;
    Signature session_sig;
} ConvContext;

typedef struct {
    uint32_t slot_nodes[WORKING_MEM_SLOTS];
    float    slot_salience[WORKING_MEM_SLOTS];
    uint8_t  n_slots;
    uint64_t last_cleared_us;
} WorkingMemory;

typedef struct {
    uint64_t  timestamp_us;
    char      query   [512];
    char      response[1024];
    float     confidence;
    uint32_t  key_nodes[16];
    uint8_t   n_key;
    LangID    lang;
    float     reward;
} Episode;

typedef struct {
    Episode  *episodes;
    uint32_t  n_episodes, cap, head;
    char      index_path[256];
} EpisodicMemory;

typedef struct {
    uint32_t  query_id;
    float     reward;
    uint64_t  timestamp_us;
    uint32_t  active_nodes[128];
    uint8_t   n_active;
} FeedbackEntry4;

typedef struct {
    float    bucket_sum  [CALIB_BUCKETS];
    float    bucket_pos  [CALIB_BUCKETS];
    uint32_t bucket_n    [CALIB_BUCKETS];
    float    calibrated  [CALIB_BUCKETS];
    bool     fitted;
} CalibTable;

typedef struct {
    float   *scores;
    uint32_t n_scores;
    uint32_t last_run_query;
    float    damping;
} PageRankState;

/* ═══════════════════════════════════════════════════════
 *  HNSW (unchanged from v3)
 * ═══════════════════════════════════════════════════════ */
typedef struct HNSWNode {
    uint32_t  node_id;
    uint32_t *links[HNSW_MAX_LAYERS];
    uint8_t   n_links[HNSW_MAX_LAYERS];
    uint8_t   link_cap[HNSW_MAX_LAYERS];
    int8_t    max_layer;
} HNSWNode;

typedef struct {
    HNSWNode *nodes;
    uint32_t  n_nodes, cap, entry_point;
    int8_t    max_layer;
    uint32_t  ef_construction, ef_search;
    pthread_rwlock_t lock;
} HNSWIndex;

/* ═══════════════════════════════════════════════════════
 *  LLM Context (unchanged from v3)
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    void  *handle;
    bool   available;
    char   model_path[512], model_name[64];
    int    n_ctx, n_threads;
    float  temperature, top_p, repeat_penalty;
    int    top_k, max_new_tokens;
    struct {
        NodeType domain;
        char system[512], user_prefix[128], user_suffix[64];
    } templates[PROMPT_TEMPLATE_CAP];
    int n_templates;
    uint64_t total_tokens_generated;
    double   avg_tokens_per_sec;
    uint32_t n_calls, n_grounding_corrections;
} LLMContext;

/* ═══════════════════════════════════════════════════════
 *  WAL (unchanged from v3)
 * ═══════════════════════════════════════════════════════ */
typedef enum {
    WAL_ADD_NODE=1, WAL_ADD_EDGE, WAL_UPDATE_WEIGHT,
    WAL_PRUNE_NODE, WAL_UPDATE_SIG
} WALOpType;

typedef struct {
    WALOpType op;
    uint32_t  id_a, id_b;
    float     value;
    uint8_t   payload[256];
    uint32_t  payload_len;
    uint64_t  timestamp_us;
    uint32_t  checksum;
} WALEntry;

typedef struct {
    FILE    *file;
    char     path[512];
    uint32_t n_entries, bytes_written;
    pthread_mutex_t lock;
} WAL;

/* ═══════════════════════════════════════════════════════
 *  Stats (extended for v4)
 * ═══════════════════════════════════════════════════════ */
typedef struct {
    uint32_t total_nodes, active_nodes, pruned_nodes;
    uint32_t total_edges, auto_nodes, fast_ring_count;
    uint32_t nodes_by_lang[LANG_COUNT];
    uint64_t total_queries, total_prunes;
    double   avg_latency_ms, avg_llm_latency_ms;
    float    avg_entropy, avg_confidence, sparsity, feedback_score;
    uint32_t hnsw_hits, hnsw_misses, llm_calls, grounding_corrections;
    uint64_t hebbian_updates, pagerank_runs;
    float    avg_pagerank;
    uint32_t selfgrow_count;
    /* v4-specific */
    uint32_t ae_runs, ae_nodes_pruned, ae_nodes_promoted;
    uint32_t vae_candidates_generated, vae_candidates_accepted;
    uint32_t spectral_runs;
    float    avg_pacbayes_bound;
    float    avg_msg_entropy;
} SRHNStats4;

/* ═══════════════════════════════════════════════════════
 *  Main Network  (v4)
 * ═══════════════════════════════════════════════════════ */
typedef struct SRHNNetwork4 {
    /* Core graph */
    SRHNNode4   *nodes;
    SRHNEdge4   *edges;
    uint32_t     n_nodes, n_edges, node_cap, edge_cap;

    /* Indices */
    HNSWIndex    hnsw;

    /* Knowledge */
    Lexicon      lexicon;
    PhysicsLaw  *physics_laws;
    uint32_t     n_physics_laws;

    /* Memory */
    ConvContext   context;
    WorkingMemory working_mem;
    EpisodicMemory episodic;
    FeedbackEntry4 feedback_buf[FEEDBACK_RING];
    uint32_t       feedback_head, feedback_count;

    /* Learning */
    PageRankState pagerank;
    CalibTable    calib;

    /* LLM */
    LLMContext    llm;

    /* Persistence */
    WAL           wal;

    /* ── v4 research modules ── */
    MsgPassState  msgpass;    /* [R2] */
    AutoEncoder   autoenc;    /* [R4] */
    HypergraphVAE vae;        /* [R6] */
    PACBayesState pacbayes;   /* [R7] */
    SpectralState spectral;   /* [R8] */

    /* VAE candidate queue */
    VAECandidate  vae_candidates[32];
    uint8_t       n_vae_candidates;

    /* Runtime */
    float        global_time;
    uint32_t     query_counter;
    uint32_t     selfgrow_count;
    float        prune_interval;
    bool         selfgrow_enabled;
    float        selfgrow_thresh;
    uint64_t     rng_state;
    SRHNError    last_error;

    /* Config flags (enable/disable research modules) */
    bool         use_multisem;    /* [R1] on by default */
    bool         use_msgpass;     /* [R2] on by default */
    bool         use_hyperedge_attn; /* [R3] on by default */
    bool         use_autoenc;     /* [R4] on by default */
    bool         use_temporal;    /* [R5] on by default */
    bool         use_vae;         /* [R6] off by default (needs user confirmation) */
    bool         use_pacbayes;    /* [R7] on by default */
    bool         use_spectral;    /* [R8] on by default */

    SRHNStats4   stats;

    pthread_rwlock_t nodes_lock;
    pthread_rwlock_t edges_lock;
    pthread_mutex_t  query_lock;
} SRHNNetwork4;

/* ═══════════════════════════════════════════════════════
 *  Inline utilities
 * ═══════════════════════════════════════════════════════ */
static inline float srhn4_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static inline float srhn4_dot(const float *a, const float *b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
static inline float srhn4_relu(float x) { return x > 0.f ? x : 0.f; }

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_core4.c
 * ═══════════════════════════════════════════════════════ */
SRHNNetwork4 *srhn4_create(void);
void          srhn4_destroy(SRHNNetwork4 *net);
void          srhn4_reset(SRHNNetwork4 *net);
uint32_t      srhn4_add_node(SRHNNetwork4 *net, const char *label, NodeType type, LangID lang);
uint32_t      srhn4_add_concept(SRHNNetwork4 *net, const char *text);
void          srhn4_connect(SRHNNetwork4 *net, uint32_t a, uint32_t b, float weight, EdgeType type);
uint32_t      srhn4_add_edge(SRHNNetwork4 *net, uint32_t *nids, uint8_t n, float strength, EdgeType type, const char *relation);
SRHNResult4   srhn4_query(SRHNNetwork4 *net, const char *input);
void          srhn4_feedback(SRHNNetwork4 *net, uint32_t query_id, float reward);
SRHNStats4    srhn4_get_stats(SRHNNetwork4 *net);
void          srhn4_print_stats(SRHNNetwork4 *net);
const char   *srhn4_error_str(SRHNError err);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_embed4.c
 * ═══════════════════════════════════════════════════════ */
bool  srhn4_embed_load_glove   (SRHNNetwork4 *net, const char *path);
bool  srhn4_embed_load_fasttext(SRHNNetwork4 *net, const char *path);
void  srhn4_embed_text         (SRHNNetwork4 *net, const char *text, LangID lang, Signature *out);
float srhn4_resonance          (const Signature *a, const Signature *b);
void  srhn4_sig_blend          (Signature *dst, const Signature *src, float alpha);

/* [R1] Multi-semantic */
void  srhn4_orthogonalize_views(Signature *sig);
float srhn4_multisem_resonance (const Signature *a, const Signature *b);
void  srhn4_multisem_update_weights(Signature *a, const Signature *b, float reward, float lr);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_hnsw4.c (unchanged from v3)
 * ═══════════════════════════════════════════════════════ */
bool srhn4_hnsw_init  (HNSWIndex *idx, uint32_t init_cap);
void srhn4_hnsw_free  (HNSWIndex *idx);
bool srhn4_hnsw_insert(HNSWIndex *idx, uint32_t node_id, const Signature *sig, SRHNNetwork4 *net);
int  srhn4_hnsw_search(HNSWIndex *idx, const Signature *query, uint32_t ef,
                        uint32_t *result_ids, float *result_dists, int k, SRHNNetwork4 *net);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_reasoning4.c
 * ═══════════════════════════════════════════════════════ */
void srhn4_propagate         (SRHNNetwork4 *net, uint32_t seed, float init_spike, SRHNResult4 *result);
void srhn4_build_chains      (SRHNNetwork4 *net, SRHNResult4 *result);
void srhn4_find_contradictions(SRHNNetwork4 *net, SRHNResult4 *result);
void srhn4_causal_path       (SRHNNetwork4 *net, uint32_t from, uint32_t to, ReasoningChain *out);
void srhn4_eval_hyperedges   (SRHNNetwork4 *net, SRHNResult4 *result);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_msgpass.c  [R2]
 * ═══════════════════════════════════════════════════════ */
bool srhn4_msgpass_init   (MsgPassState *mp, uint32_t n_nodes, uint32_t n_edges);
void srhn4_msgpass_free   (MsgPassState *mp);
void srhn4_msgpass_resize (MsgPassState *mp, uint32_t new_nodes, uint32_t new_edges);
void srhn4_msgpass_run    (SRHNNetwork4 *net, SRHNResult4 *result);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_hyperedge_attn.c  [R3]
 * ═══════════════════════════════════════════════════════ */
void  srhn4_hyperedge_attn_run (SRHNNetwork4 *net, SRHNResult4 *result);
float srhn4_edge_attn_entropy  (SRHNEdge4 *edge);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_autoenc.c  [R4]
 * ═══════════════════════════════════════════════════════ */
void  srhn4_ae_init    (AutoEncoder *ae, float lr);
void  srhn4_ae_train   (SRHNNetwork4 *net, uint32_t n_iters);
void  srhn4_ae_run     (SRHNNetwork4 *net);   /* encode all → check recon error */
float srhn4_ae_reconstruct_error(AutoEncoder *ae, const float *sig);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_temporal.c  [R5]
 * ═══════════════════════════════════════════════════════ */
float srhn4_temporal_weight    (float base_weight, uint64_t last_fired_us);
void  srhn4_temporal_update_edge(SRHNNetwork4 *net, uint32_t node_a, uint8_t edge_idx);
void  srhn4_temporal_decay_pass(SRHNNetwork4 *net);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_vae.c  [R6]
 * ═══════════════════════════════════════════════════════ */
void  srhn4_vae_init    (HypergraphVAE *vae, float lr);
void  srhn4_vae_train   (SRHNNetwork4 *net, uint32_t n_epochs);
void  srhn4_vae_generate(SRHNNetwork4 *net, const char *domain_hint, uint32_t n_candidates);
bool  srhn4_vae_accept  (SRHNNetwork4 *net, uint8_t candidate_idx);
void  srhn4_vae_reject  (SRHNNetwork4 *net, uint8_t candidate_idx);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_pacbayes.c  [R7]
 * ═══════════════════════════════════════════════════════ */
void  srhn4_pacbayes_update(SRHNNetwork4 *net);
float srhn4_pacbayes_bound (SRHNNetwork4 *net, float empirical_risk);
void  srhn4_pacbayes_apply_to_result(SRHNNetwork4 *net, SRHNResult4 *result);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_spectral.c  [R8]
 * ═══════════════════════════════════════════════════════ */
bool srhn4_spectral_init   (SpectralState *ss, uint32_t n_nodes);
void srhn4_spectral_free   (SpectralState *ss);
void srhn4_spectral_compute(SRHNNetwork4 *net);
void srhn4_spectral_rescore_seeds(SRHNNetwork4 *net, uint32_t *seed_ids,
                                   float *seed_dists, int n_seeds);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_feedback4.c
 * ═══════════════════════════════════════════════════════ */
void  srhn4_apply_feedback  (SRHNNetwork4 *net);
void  srhn4_hebbian_update  (SRHNNetwork4 *net, FeedbackEntry4 *entry);
void  srhn4_contrastive_sig (SRHNNetwork4 *net, FeedbackEntry4 *entry);
void  srhn4_run_pagerank    (SRHNNetwork4 *net);
void  srhn4_calibrate       (SRHNNetwork4 *net, float predicted, bool was_correct);
float srhn4_calibrated_conf (SRHNNetwork4 *net, float raw);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_memory4.c
 * ═══════════════════════════════════════════════════════ */
void srhn4_working_mem_insert(SRHNNetwork4 *net, uint32_t node_id, float salience);
void srhn4_working_mem_clear (SRHNNetwork4 *net);
void srhn4_working_mem_apply (SRHNNetwork4 *net);
void srhn4_episodic_store    (SRHNNetwork4 *net, SRHNResult4 *result, const char *query, float reward);
void srhn4_episodic_recall   (SRHNNetwork4 *net, const Signature *qsig, int k, Episode **out, int *n_out);
bool srhn4_episodic_init     (SRHNNetwork4 *net, uint32_t cap);
void srhn4_episodic_free     (SRHNNetwork4 *net);
void srhn4_ctx_blend_query   (SRHNNetwork4 *net, Signature *eff, const Signature *qsig);
void srhn4_ctx_update        (SRHNNetwork4 *net, const char *query, SRHNResult4 *result, const Signature *qsig);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_persist4.c
 * ═══════════════════════════════════════════════════════ */
bool          srhn4_save       (SRHNNetwork4 *net, const char *path);
SRHNNetwork4 *srhn4_load       (const char *path);
bool          srhn4_wal_open   (SRHNNetwork4 *net, const char *path);
void          srhn4_wal_close  (SRHNNetwork4 *net);
bool          srhn4_wal_write  (SRHNNetwork4 *net, WALOpType op, uint32_t a, uint32_t b, float val, const void *payload, uint32_t plen);
bool          srhn4_wal_replay (SRHNNetwork4 *net, const char *path);
void          srhn4_export_json(SRHNNetwork4 *net, const char *path);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_utils4.c
 * ═══════════════════════════════════════════════════════ */
uint64_t    srhn4_timestamp_us(void);
uint64_t    srhn4_rand        (SRHNNetwork4 *net);
float       srhn4_randf       (SRHNNetwork4 *net);
LangID      srhn4_detect_lang (const char *text);
const char *srhn4_lang_name   (LangID lang);
void        srhn4_str_lower   (char *s);
uint32_t    srhn4_selfgrow    (SRHNNetwork4 *net, const char *text, LangID lang);
void        srhn4_selfgrow_from_query(SRHNNetwork4 *net, const char *q, float best_res);
void        srhn4_prune_pass  (SRHNNetwork4 *net);
void        srhn4_promote_fast(SRHNNetwork4 *net, uint32_t nid);
void        srhn4_load_physics(SRHNNetwork4 *net);
void        srhn4_load_code   (SRHNNetwork4 *net);
void        srhn4_ingest_document(SRHNNetwork4 *net, const char *text, uint32_t source_id);
int         srhn4_tok_simple  (const char *text, char tokens[][MAX_WORD_LEN], int max_tok);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_llm4.c
 * ═══════════════════════════════════════════════════════ */
bool srhn4_llm_init         (LLMContext *llm, const char *path, int n_ctx, int threads);
void srhn4_llm_free         (LLMContext *llm);
bool srhn4_llm_generate     (LLMContext *llm, const char *prompt, char *out, int max_out);
void srhn4_build_context    (SRHNNetwork4 *net, SRHNResult4 *result, const char *query, char *out, int sz);
void srhn4_answer           (SRHNNetwork4 *net, const char *query, SRHNResult4 *result);
bool srhn4_ground_response  (SRHNNetwork4 *net, SRHNResult4 *result);
void srhn4_llm_feedback_grow(SRHNNetwork4 *net, const char *response, SRHNResult4 *result);
void auto_prompt_templates4 (LLMContext *llm);

/* ═══════════════════════════════════════════════════════
 *  Forward declarations — srhn_api4.c
 * ═══════════════════════════════════════════════════════ */
typedef struct SRHNServer4 SRHNServer4;
SRHNServer4 *srhn4_server_create(SRHNNetwork4 *net, int port);
void         srhn4_server_run  (SRHNServer4 *srv);
void         srhn4_server_stop (SRHNServer4 *srv);
void         srhn4_server_free (SRHNServer4 *srv);

/* Extra decls needed across modules */
void  srhn4_ctx_update(SRHNNetwork4 *net, const char *query, SRHNResult4 *result, const Signature *sig);

#endif /* SRHN_V4_H */
