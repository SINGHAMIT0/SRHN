/*
 * srhn_v3_train.h  —  SRHN v3 Training Module Header
 * =====================================================
 * Training pipeline for SRHN v3: 8 domains, 120+ samples,
 * online Hebbian learning, benchmark suite, CSV/JSON export.
 */
#pragma once
#ifndef SRHN_V4_TRAIN_H
#define SRHN_V4_TRAIN_H

#define _POSIX_C_SOURCE 200809L
#include "srhn_v4.h"

/* ── Constants ───────────────────────────────────────────────── */
#define TRAIN_MAX_SAMPLES     512
#define TRAIN_MAX_CONCEPTS    12
#define TRAIN_MAX_KEYWORDS    8
#define TRAIN_LOG_SIZE        4096
#define TRAIN_DEFAULT_EPOCHS  10
#define TRAIN_DEFAULT_LR      0.04f

/* ── Per-sample metrics ──────────────────────────────────────── */
typedef struct {
    uint32_t epoch;
    uint32_t sample_idx;
    float    loss;
    float    accuracy;
    uint32_t nodes_added;
    uint32_t edges_added;
    double   elapsed_ms;
    float    confidence;
    float    best_resonance;
    bool     used_llm;
} TrainMetrics;

/* ── Training sample ─────────────────────────────────────────── */
typedef struct {
    char     query   [256];
    char     domain  [32];
    char     concepts[TRAIN_MAX_CONCEPTS][MAX_WORD_LEN];
    char     keywords[TRAIN_MAX_KEYWORDS][MAX_WORD_LEN];
    char     causal_rule[256];
    int      n_concepts;
    int      n_keywords;
    LangID   lang;
    float    difficulty;
    float    expected_confidence;
    NodeType node_type;           /* preferred node type for concepts */
} TrainSample;

/* ── Training dataset ────────────────────────────────────────── */
typedef struct {
    TrainSample samples[TRAIN_MAX_SAMPLES];
    uint32_t    n_samples;
    char        name[64];
} TrainDataset;

/* ── Training state ──────────────────────────────────────────── */
typedef struct {
    TrainDataset dataset;
    TrainMetrics history[TRAIN_LOG_SIZE];
    uint32_t     history_len;

    float    learning_rate;
    uint32_t epoch;
    uint32_t total_tested;
    uint32_t correct;
    float    total_loss;
    float    total_acc;
    uint32_t total_added;
    uint32_t llm_calls;

    bool     verbose;
    bool     running;

    /* Domain accuracy tracking */
    struct {
        char     name[32];
        uint32_t tested;
        uint32_t correct;
        float    avg_conf;
    } domain_stats[16];
    int n_domains;
} TrainState;

/* ── Public API ──────────────────────────────────────────────── */
void          srhn_train_init         (TrainState *ts, float lr);
void          srhn_train_load_builtin (TrainDataset *ds);
TrainMetrics  srhn_train_step         (SRHNNetwork4 *net, TrainState *ts,
                                        const TrainSample *sample);
void          srhn_train_epoch        (SRHNNetwork4 *net, TrainState *ts);
void          srhn_train_run          (SRHNNetwork4 *net, TrainState *ts, uint32_t n_epochs);
void          srhn_train_benchmark    (SRHNNetwork4 *net, TrainState *ts);
float         srhn_train_eval         (SRHNNetwork4 *net, const char *query,
                                        const char **expected, int n_expected);
float         srhn_train_loss         (SRHNNetwork4 *net, const TrainSample *sample);
void          srhn_train_online       (SRHNNetwork4 *net, TrainState *ts);
void          srhn_train_ingest_file  (SRHNNetwork4 *net, const char *path, LangID lang);
void          srhn_train_add_sample   (TrainDataset *ds,
                                        const char *query,
                                        const char **concepts, int n_concepts,
                                        const char *causal_rule,
                                        const char **keywords, int n_keywords,
                                        const char *domain, LangID lang,
                                        float difficulty);
void          srhn_train_print_summary(TrainState *ts);
void          srhn_train_export_csv   (TrainState *ts, const char *path);
void          srhn_train_export_json  (SRHNNetwork4 *net, TrainState *ts,
                                        const char *path);

#endif /* SRHN_V4_TRAIN_H */
