/*
 * srhn_embed.c  —  Dense vector embeddings for SRHN v3
 *
 * Supports:
 *   1. GloVe binary format (300-dim)
 *   2. FastText binary format (300-dim)
 *   3. Fallback: char n-gram hashing (no file needed)
 *
 * Text embedding strategy:
 *   - BPE tokenize input
 *   - Look up each token in vocab → get 300-dim vector
 *   - IDF-weighted mean pooling
 *   - Project 300 → EMBED_DIM (384) via random orthogonal matrix (seeded)
 *   - L2 normalize
 */

#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <ctype.h>

/* ── Internal vocab entry ────────────────────────────────────── */
#define EMBED_RAW_DIM   300
#define EMBED_HASH_SIZE 131072   /* must be power of 2 */

typedef struct {
    char    word[MAX_WORD_LEN];
    float   vec[EMBED_RAW_DIM];
    float   idf;
    uint32_t hash;
} EmbedEntry;

typedef struct {
    EmbedEntry  *entries;
    uint32_t    *buckets;    /* open-addressing hash table */
    uint32_t     n_entries;
    uint32_t     cap;
    uint32_t     hash_cap;   /* must be power of 2 */
    float        proj[EMBED_RAW_DIM][EMBED_DIM]; /* projection matrix */
    bool         proj_ready;
    bool         loaded;
} EmbedVocab;

/* Global vocab (one per network — stored in net via thread-local hack or
   we embed a pointer.  For simplicity, we use a global singleton guard) */
static EmbedVocab g_vocab = {0};
static pthread_mutex_t g_vocab_lock = PTHREAD_MUTEX_INITIALIZER;

/* ── FNV-1a hash ────────────────────────────────────────────── */
static uint32_t fnv1a(const char *s) {
    uint32_t h = 0x811c9dc5u;
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 0x01000193u;
    }
    return h;
}

/* ── Random projection matrix (seeded, stable across runs) ─── */
static float lcg_randf(uint64_t *state) {
    *state = (*state * 6364136223846793005ULL) + 1442695040888963407ULL;
    uint32_t v = (uint32_t)(*state >> 33);
    return ((float)v / (float)0xFFFFFFFFu) * 2.f - 1.f;
}

static void build_projection(EmbedVocab *v) {
    uint64_t rng = 0xDEADBEEFCAFE1234ULL;
    /* Generate random Gaussian-ish projection using Box-Muller */
    for (int i = 0; i < EMBED_RAW_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            float u1 = lcg_randf(&rng);
            float u2 = lcg_randf(&rng);
            /* Approximate normal: sum of 3 uniforms → CLT */
            float u3 = lcg_randf(&rng);
            v->proj[i][j] = (u1 + u2 + u3) / 3.f * sqrtf(3.f);
        }
        /* Normalize each output dimension */
        float norm = 0.f;
        for (int j = 0; j < EMBED_DIM; j++) norm += v->proj[i][j] * v->proj[i][j];
        norm = sqrtf(norm) + 1e-8f;
        for (int j = 0; j < EMBED_DIM; j++) v->proj[i][j] /= norm;
    }
    v->proj_ready = true;
}

/* ── Hash table operations ──────────────────────────────────── */
static uint32_t vocab_find(EmbedVocab *v, const char *word) {
    uint32_t h = fnv1a(word) & (v->hash_cap - 1);
    for (uint32_t probe = 0; probe < v->hash_cap; probe++) {
        uint32_t idx = (h + probe) & (v->hash_cap - 1);
        if (v->buckets[idx] == UINT32_MAX) return UINT32_MAX;
        if (strncmp(v->entries[v->buckets[idx]].word, word, MAX_WORD_LEN-1) == 0)
            return v->buckets[idx];
    }
    return UINT32_MAX;
}

static bool vocab_insert(EmbedVocab *v, const char *word, const float *vec) {
    if (v->n_entries >= v->cap * 7 / 8) return false; /* 87.5% load */
    uint32_t h = fnv1a(word) & (v->hash_cap - 1);
    for (uint32_t probe = 0; probe < v->hash_cap; probe++) {
        uint32_t idx = (h + probe) & (v->hash_cap - 1);
        if (v->buckets[idx] == UINT32_MAX) {
            v->buckets[idx] = v->n_entries;
            EmbedEntry *e = &v->entries[v->n_entries++];
            strncpy(e->word, word, MAX_WORD_LEN - 1);
            memcpy(e->vec, vec, EMBED_RAW_DIM * sizeof(float));
            e->idf  = 1.f;
            e->hash = fnv1a(word);
            return true;
        }
    }
    return false;
}

/* ── Load GloVe text format ─────────────────────────────────── */
static bool load_glove_text(EmbedVocab *v, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return false;

    char word[MAX_WORD_LEN];
    float vec[EMBED_RAW_DIM];
    uint32_t loaded = 0;

    while (fscanf(f, "%63s", word) == 1 && loaded < v->cap - 1) {
        bool ok = true;
        for (int i = 0; i < EMBED_RAW_DIM; i++) {
            if (fscanf(f, "%f", &vec[i]) != 1) { ok = false; break; }
        }
        if (!ok) { /* skip rest of line */
            int c; while ((c = fgetc(f)) != '\n' && c != EOF);
            continue;
        }
        /* L2-normalize the raw vector */
        float norm = 0.f;
        for (int i = 0; i < EMBED_RAW_DIM; i++) norm += vec[i] * vec[i];
        norm = sqrtf(norm) + 1e-8f;
        for (int i = 0; i < EMBED_RAW_DIM; i++) vec[i] /= norm;

        vocab_insert(v, word, vec);
        loaded++;
    }
    fclose(f);
    fprintf(stderr, "[embed] Loaded %u GloVe vectors from %s\n", loaded, path);
    return loaded > 0;
}

/* ── Load FastText binary format ────────────────────────────── */
static bool load_fasttext_bin(EmbedVocab *v, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    /* FastText binary header: rows cols */
    long n_words = 0, dim = 0;
    if (fscanf(f, "%ld %ld\n", &n_words, &dim) != 2) { fclose(f); return false; }
    if (dim != EMBED_RAW_DIM) {
        fprintf(stderr, "[embed] FastText dim=%ld, expected %d\n", dim, EMBED_RAW_DIM);
        fclose(f); return false;
    }

    char word[MAX_WORD_LEN];
    float vec[EMBED_RAW_DIM];
    uint32_t loaded = 0;

    for (long i = 0; i < n_words && loaded < (long)(v->cap - 1); i++) {
        /* Read word (space-terminated) */
        int wi = 0;
        int c;
        while ((c = fgetc(f)) != ' ' && c != EOF && wi < MAX_WORD_LEN - 1)
            word[wi++] = (char)c;
        word[wi] = '\0';
        if (c == EOF) break;

        /* Read binary float vector */
        if ((int)fread(vec, sizeof(float), EMBED_RAW_DIM, f) != EMBED_RAW_DIM) break;

        /* L2 normalize */
        float norm = 0.f;
        for (int j = 0; j < EMBED_RAW_DIM; j++) norm += vec[j] * vec[j];
        norm = sqrtf(norm) + 1e-8f;
        for (int j = 0; j < EMBED_RAW_DIM; j++) vec[j] /= norm;

        vocab_insert(v, word, vec);
        loaded++;
    }
    fclose(f);
    fprintf(stderr, "[embed] Loaded %u FastText vectors from %s\n", loaded, path);
    return loaded > 0;
}

/* ── Project 300-dim → EMBED_DIM ────────────────────────────── */
static void project_embed(EmbedVocab *v, const float *raw, float *out) {
    memset(out, 0, EMBED_DIM * sizeof(float));
    for (int i = 0; i < EMBED_RAW_DIM; i++) {
        if (raw[i] == 0.f) continue;
        for (int j = 0; j < EMBED_DIM; j++)
            out[j] += raw[i] * v->proj[i][j];
    }
}

/* ── Fallback: char n-gram hashing (no vocab needed) ────────── */
static void ngram_embed(const char *text, float *out) {
    memset(out, 0, EMBED_DIM * sizeof(float));
    int len = (int)strlen(text);
    if (len == 0) return;

    /* Trigrams */
    for (int i = 0; i < len; i++) {
        uint8_t c0 = (uint8_t)tolower((unsigned char)text[i]);
        uint8_t c1 = (i+1 < len) ? (uint8_t)tolower((unsigned char)text[i+1]) : 0;
        uint8_t c2 = (i+2 < len) ? (uint8_t)tolower((unsigned char)text[i+2]) : 0;

        uint32_t tg = ((uint32_t)c0 << 16) | ((uint32_t)c1 << 8) | c2;
        uint32_t h = tg ^ 0x811c9dc5u;
        h = (h ^ (h >> 16)) * 0x45d9f3bu;
        h = (h ^ (h >> 16)) * 0x45d9f3bu;
        h ^= h >> 16;

        int bk = (int)(h % EMBED_DIM);
        float ph = ((float)(h & 0xFFFF) / 65535.f) * 6.28318f;
        out[bk]                     += cosf(ph);
        out[(bk + 1) % EMBED_DIM]  += sinf(ph);
        out[(bk + 17) % EMBED_DIM] += cosf(ph * 2.f) * 0.5f;
    }

    /* Bigrams (higher weight) */
    for (int i = 0; i < len - 1; i++) {
        uint8_t c0 = (uint8_t)tolower((unsigned char)text[i]);
        uint8_t c1 = (uint8_t)tolower((unsigned char)text[i+1]);
        uint32_t bg = ((uint32_t)c0 << 8) | c1;
        uint32_t h = bg * 2654435761u;
        h ^= h >> 16;
        int bk = (int)(h % EMBED_DIM);
        out[bk] += 1.5f;
    }

    /* Word-level features */
    char buf[512];
    strncpy(buf, text, 511); buf[511] = '\0';
    char *sp = NULL;
    char *tok = strtok_r(buf, " \t\n\r,.!?;:\"'()[]{}", &sp);
    int wi = 0;
    while (tok && wi < 64) {
        uint32_t wh = 5381;
        for (char *p = tok; *p; p++)
            wh = ((wh << 5) + wh) ^ (uint8_t)tolower((unsigned char)*p);
        int wb = (int)(wh % EMBED_DIM);
        out[wb] += 3.f;
        out[(wb + 8)  % EMBED_DIM] += 1.5f;
        float pw = 2.f / (1.f + wi * 0.15f);
        out[(wb + 32) % EMBED_DIM] += pw;
        tok = strtok_r(NULL, " \t\n\r,.!?;:\"'()[]{}", &sp);
        wi++;
    }
}

/* ── Public API ─────────────────────────────────────────────── */

bool srhn4_embed_load_glove(SRHNNetwork4 *net, const char *bin_path) {
    (void)net;
    pthread_mutex_lock(&g_vocab_lock);

    if (!g_vocab.entries) {
        g_vocab.cap       = 400000;
        g_vocab.hash_cap  = EMBED_HASH_SIZE * 4; /* 524288 */
        g_vocab.entries   = (EmbedEntry *)calloc(g_vocab.cap, sizeof(EmbedEntry));
        g_vocab.buckets   = (uint32_t   *)malloc(g_vocab.hash_cap * sizeof(uint32_t));
        if (!g_vocab.entries || !g_vocab.buckets) {
            pthread_mutex_unlock(&g_vocab_lock);
            return false;
        }
        memset(g_vocab.buckets, 0xFF, g_vocab.hash_cap * sizeof(uint32_t));
        build_projection(&g_vocab);
    }

    bool ok = load_glove_text(&g_vocab, bin_path);
    if (ok) g_vocab.loaded = true;

    pthread_mutex_unlock(&g_vocab_lock);
    return ok;
}

bool srhn4_embed_load_fasttext(SRHNNetwork4 *net, const char *bin_path) {
    (void)net;
    pthread_mutex_lock(&g_vocab_lock);

    if (!g_vocab.entries) {
        g_vocab.cap      = 2000000;
        g_vocab.hash_cap = EMBED_HASH_SIZE * 16; /* 2097152 */
        g_vocab.entries  = (EmbedEntry *)calloc(g_vocab.cap, sizeof(EmbedEntry));
        g_vocab.buckets  = (uint32_t   *)malloc(g_vocab.hash_cap * sizeof(uint32_t));
        if (!g_vocab.entries || !g_vocab.buckets) {
            pthread_mutex_unlock(&g_vocab_lock);
            return false;
        }
        memset(g_vocab.buckets, 0xFF, g_vocab.hash_cap * sizeof(uint32_t));
        build_projection(&g_vocab);
    }

    bool ok = load_fasttext_bin(&g_vocab, bin_path);
    if (ok) g_vocab.loaded = true;

    pthread_mutex_unlock(&g_vocab_lock);
    return ok;
}

/*
 * srhn4_embed_text — compute EMBED_DIM signature for a text string.
 *
 * Strategy:
 *   1. Tokenize into words (simple whitespace + punct split)
 *   2. Look up each word in vocab → get raw 300-dim vector
 *   3. IDF-weight each vector
 *   4. Mean pool
 *   5. Project 300 → EMBED_DIM
 *   6. If vocab unavailable: fallback to n-gram hashing
 *   7. L2 normalize
 */
void srhn4_embed_text(SRHNNetwork4 *net, const char *text, LangID lang, Signature *out) {
    (void)net;
    memset(out->vec, 0, sizeof(out->vec));
    out->hash = fnv1a(text);
    out->lang = lang;

    if (!text || !*text) { out->magnitude = 0.f; return; }

    bool use_vocab = g_vocab.loaded && g_vocab.entries;

    if (use_vocab) {
        /* Tokenize */
        char buf[1024];
        strncpy(buf, text, 1023); buf[1023] = '\0';

        float pooled[EMBED_RAW_DIM];
        memset(pooled, 0, sizeof(pooled));
        float total_weight = 0.f;

        /* Simple word tokenization for embedding lookup */
        char *sp = NULL;
        char *tok = strtok_r(buf, " \t\n\r,.!?;:\"'()[]{}<>", &sp);
        int widx = 0;
        while (tok && widx < 128) {
            /* Try original case, then lowercase */
            uint32_t idx = vocab_find(&g_vocab, tok);
            if (idx == UINT32_MAX) {
                char lower[MAX_WORD_LEN];
                strncpy(lower, tok, MAX_WORD_LEN - 1); lower[MAX_WORD_LEN-1] = '\0';
                for (char *p = lower; *p; p++) *p = (char)tolower((unsigned char)*p);
                idx = vocab_find(&g_vocab, lower);
            }

            if (idx != UINT32_MAX) {
                float w = g_vocab.entries[idx].idf + 0.1f;
                /* Position weight: earlier tokens matter slightly more */
                float pos_w = 1.f / (1.f + widx * 0.05f);
                w *= pos_w;
                for (int d = 0; d < EMBED_RAW_DIM; d++)
                    pooled[d] += g_vocab.entries[idx].vec[d] * w;
                total_weight += w;
            }
            tok = strtok_r(NULL, " \t\n\r,.!?;:\"'()[]{}<>", &sp);
            widx++;
        }

        if (total_weight > 0.f) {
            for (int d = 0; d < EMBED_RAW_DIM; d++) pooled[d] /= total_weight;
            project_embed(&g_vocab, pooled, out->vec);
        } else {
            /* All OOV words — fallback to n-gram */
            ngram_embed(text, out->vec);
        }
    } else {
        /* No vocab loaded — pure n-gram fallback */
        ngram_embed(text, out->vec);
    }

    /* Language-specific scaling */
    float scale = 1.f;
    if (lang == LANG_HINDI || lang == LANG_ARABIC) scale = 1.15f;
    if (lang == LANG_CODE)  scale = 0.92f;
    if (lang == LANG_MIXED) scale = 0.95f;
    for (int i = 0; i < EMBED_DIM; i++) out->vec[i] *= scale;

    /* L2 normalize */
    float mag = 0.f;
    for (int i = 0; i < EMBED_DIM; i++) mag += out->vec[i] * out->vec[i];
    mag = sqrtf(mag);
    out->magnitude = mag;
    if (mag > 1e-8f)
        for (int i = 0; i < EMBED_DIM; i++) out->vec[i] /= mag;
    out->magnitude = 1.f; /* after normalization */
}

/*
 * srhn4_resonance — cosine similarity with fast pre-filter.
 *
 * Returns [0, 1].  Cross-language discount applied.
 */
float srhn4_resonance(const Signature *a, const Signature *b) {
    if (!a || !b) return 0.f;

    /* Fast pre-filter: if hash XOR popcount > threshold, skip */
    uint32_t diff = (uint32_t)__builtin_popcount(a->hash ^ b->hash);
    if (diff > 28) return 0.f;

    /* Cross-language soft penalty */
    float cross = 1.f;
    if (a->lang != b->lang &&
        a->lang != LANG_UNKNOWN && b->lang != LANG_UNKNOWN &&
        a->lang != LANG_MIXED   && b->lang != LANG_MIXED) {
        /* English ↔ Code: slight discount */
        if ((a->lang == LANG_CODE) != (b->lang == LANG_CODE))
            cross = 0.90f;
        else
            cross = 0.82f;
    }

    float dot = srhn4_dot(a->vec, b->vec, EMBED_DIM);
    return srhn4_clampf(dot * cross, 0.f, 1.f);
}

/*
 * srhn4_sig_blend — exponential moving average of two signatures.
 * alpha=0 → keep dst unchanged; alpha=1 → become src.
 */
void srhn4_sig_blend(Signature *dst, const Signature *src, float alpha) {
    if (!dst || !src || alpha <= 0.f) return;
    alpha = srhn4_clampf(alpha, 0.f, 1.f);
    float inv = 1.f - alpha;
    float mag = 0.f;
    for (int i = 0; i < EMBED_DIM; i++) {
        dst->vec[i] = inv * dst->vec[i] + alpha * src->vec[i];
        mag += dst->vec[i] * dst->vec[i];
    }
    mag = sqrtf(mag);
    dst->magnitude = mag;
    if (mag > 1e-8f)
        for (int i = 0; i < EMBED_DIM; i++) dst->vec[i] /= mag;
    dst->magnitude = 1.f;
    /* hash is XOR blend — approximate */
    dst->hash ^= src->hash;
}
