/*
 * srhn_tokenizer.c  —  BPE Tokenizer for SRHN v3
 *
 * Implements Byte Pair Encoding tokenization compatible with GPT-2 vocab.
 * Falls back gracefully to whitespace + punct tokenizer when vocab not loaded.
 *
 * File format expected (vocab.bpe):
 *   Line 0:    #version: 0.2
 *   Lines 1-N: merge rules "ab cd" (most frequent pair first)
 *
 * vocab.json: {"token": id, ...}  (standard GPT-2 format)
 */

#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

/* ── Internal types ─────────────────────────────────────────── */
#define BPE_MAX_MERGES   50000
#define BPE_MAX_VOCAB    65536
#define BPE_HASH_SIZE    131072  /* must be power of 2 */
#define BPE_MAX_WORD     256     /* max BPE symbols per word */

typedef struct {
    char     left [MAX_TOKEN_LEN];
    char     right[MAX_TOKEN_LEN];
    char     result[MAX_TOKEN_LEN];
    uint32_t rank;               /* lower = higher priority */
} BPEMerge;

typedef struct {
    char     token[MAX_TOKEN_LEN];
    uint16_t id;
} VocabEntry;

typedef struct BPETokenizer4 BPETokenizer;
struct BPETokenizer4 {
    BPEMerge  *merges;
    uint32_t   n_merges;

    VocabEntry *vocab;
    uint32_t   *vocab_hash;   /* token → id hash map */
    uint32_t    n_vocab;
    uint32_t    vocab_hash_cap;

    /* Reverse vocab: id → token string */
    char      (*id_to_token)[MAX_TOKEN_LEN];

    /* Byte-level encoding table (GPT-2 style) */
    char       byte_enc[256][8];

    bool       loaded;
    bool       has_vocab;
};

/* ── FNV hash ───────────────────────────────────────────────── */
static uint32_t fnv_tok(const char *s) {
    uint32_t h = 0x811c9dc5u;
    for (; *s; s++) { h ^= (uint8_t)*s; h *= 0x01000193u; }
    return h;
}

/* ── Hash map operations ────────────────────────────────────── */
static uint16_t vocab_lookup(BPETokenizer *tok, const char *token) {
    if (!tok->has_vocab) return UINT16_MAX;
    uint32_t h = fnv_tok(token) & (tok->vocab_hash_cap - 1);
    for (uint32_t p = 0; p < tok->vocab_hash_cap; p++) {
        uint32_t idx = (h + p) & (tok->vocab_hash_cap - 1);
        if (tok->vocab_hash[idx] == UINT32_MAX) return UINT16_MAX;
        uint32_t vi = tok->vocab_hash[idx];
        if (strncmp(tok->vocab[vi].token, token, MAX_TOKEN_LEN-1) == 0)
            return tok->vocab[vi].id;
    }
    return UINT16_MAX;
}

static void vocab_insert_tok(BPETokenizer *tok, const char *token, uint16_t id) {
    if (tok->n_vocab >= BPE_MAX_VOCAB - 1) return;
    uint32_t h = fnv_tok(token) & (tok->vocab_hash_cap - 1);
    for (uint32_t p = 0; p < tok->vocab_hash_cap; p++) {
        uint32_t idx = (h + p) & (tok->vocab_hash_cap - 1);
        if (tok->vocab_hash[idx] == UINT32_MAX) {
            tok->vocab_hash[idx] = tok->n_vocab;
            strncpy(tok->vocab[tok->n_vocab].token, token, MAX_TOKEN_LEN-1);
            tok->vocab[tok->n_vocab].id = id;
            if (tok->id_to_token && id < BPE_MAX_VOCAB)
                strncpy(tok->id_to_token[id], token, MAX_TOKEN_LEN-1);
            tok->n_vocab++;
            return;
        }
    }
}

/* ── Build GPT-2 byte encoder ───────────────────────────────── */
static void build_byte_encoder(BPETokenizer *tok) {
    /*
     * GPT-2 maps bytes to unicode characters to avoid control chars.
     * For our purposes we just use printable bytes directly.
     */
    for (int i = 0; i < 256; i++) {
        if (isprint(i) && i != ' ') {
            tok->byte_enc[i][0] = (char)i;
            tok->byte_enc[i][1] = '\0';
        } else {
            /* Encode as Ġ prefix pattern for GPT-2 compat */
            snprintf(tok->byte_enc[i], 8, "<%02X>", i);
        }
    }
    /* Space → Ġ (represented as "Ġ" in GPT-2, we use "_" for simplicity) */
    tok->byte_enc[' '][0] = '_';
    tok->byte_enc[' '][1] = '\0';
}

/* ── BPE encode a single word ───────────────────────────────── */
static int bpe_encode_word(BPETokenizer *tok, const char *word,
                            char symbols[][MAX_TOKEN_LEN], int max_sym) {
    /* Start with character-level symbols */
    int n = 0;
    const char *p = word;
    while (*p && n < max_sym - 1) {
        /* UTF-8 aware: extract one character */
        uint8_t b = (uint8_t)*p;
        int char_bytes = 1;
        if ((b & 0xE0) == 0xC0)      char_bytes = 2;
        else if ((b & 0xF0) == 0xE0) char_bytes = 3;
        else if ((b & 0xF8) == 0xF0) char_bytes = 4;

        int si = 0;
        for (int i = 0; i < char_bytes && *p && si < MAX_TOKEN_LEN-1; i++, p++)
            symbols[n][si++] = *p;
        symbols[n][si] = '\0';
        n++;
    }
    if (n == 0) return 0;

    /* Mark word boundary */
    if (n > 0) {
        /* Append </w> to last symbol for word boundary */
        char last[MAX_TOKEN_LEN];
        snprintf(last, MAX_TOKEN_LEN, "%s</w>", symbols[n-1]);
        strncpy(symbols[n-1], last, MAX_TOKEN_LEN-1);
    }

    if (!tok->loaded || tok->n_merges == 0) return n;

    /* Apply BPE merges iteratively */
    bool changed = true;
    int max_iter = tok->n_merges;
    for (int iter = 0; iter < max_iter && changed && n > 1; iter++) {
        changed = false;
        uint32_t best_rank = UINT32_MAX;
        int best_pos = -1;

        /* Find pair with lowest merge rank */
        for (int i = 0; i < n - 1; i++) {
            for (uint32_t m = 0; m < tok->n_merges; m++) {
                if (strncmp(tok->merges[m].left,  symbols[i],   MAX_TOKEN_LEN-1) == 0 &&
                    strncmp(tok->merges[m].right, symbols[i+1], MAX_TOKEN_LEN-1) == 0) {
                    if (tok->merges[m].rank < best_rank) {
                        best_rank = tok->merges[m].rank;
                        best_pos  = i;
                    }
                    break; /* merges are sorted by rank, first match is best */
                }
            }
        }

        if (best_pos < 0) break;

        /* Merge symbols[best_pos] + symbols[best_pos+1] */
        char merged[MAX_TOKEN_LEN];
        snprintf(merged, MAX_TOKEN_LEN, "%s%s", symbols[best_pos], symbols[best_pos+1]);

        strncpy(symbols[best_pos], merged, MAX_TOKEN_LEN-1);
        /* Shift remaining symbols left */
        for (int i = best_pos + 1; i < n - 1; i++)
            strncpy(symbols[i], symbols[i+1], MAX_TOKEN_LEN-1);
        n--;
        changed = true;
    }

    return n;
}

/* ── Public API ─────────────────────────────────────────────── */

BPETokenizer *srhn4_tok_create(void) {
    BPETokenizer *tok = (BPETokenizer *)calloc(1, sizeof(BPETokenizer));
    if (!tok) return NULL;

    tok->merges        = (BPEMerge    *)calloc(BPE_MAX_MERGES, sizeof(BPEMerge));
    tok->vocab         = (VocabEntry  *)calloc(BPE_MAX_VOCAB,  sizeof(VocabEntry));
    tok->id_to_token   = (char (*)[MAX_TOKEN_LEN])calloc(BPE_MAX_VOCAB, MAX_TOKEN_LEN);
    tok->vocab_hash_cap = BPE_HASH_SIZE * 2;
    tok->vocab_hash    = (uint32_t    *)malloc(tok->vocab_hash_cap * sizeof(uint32_t));

    if (!tok->merges || !tok->vocab || !tok->id_to_token || !tok->vocab_hash) {
        free(tok->merges); free(tok->vocab);
        free(tok->id_to_token); free(tok->vocab_hash);
        free(tok);
        return NULL;
    }

    memset(tok->vocab_hash, 0xFF, tok->vocab_hash_cap * sizeof(uint32_t));
    build_byte_encoder(tok);
    return tok;
}

void srhn4_tok_destroy(BPETokenizer *tok) {
    if (!tok) return;
    free(tok->merges);
    free(tok->vocab);
    free(tok->id_to_token);
    free(tok->vocab_hash);
    free(tok);
}

/*
 * srhn4_tok_load — load BPE merge rules from GPT-2 style vocab.bpe file.
 * Format: first line is comment (#version: 0.2), then "tok1 tok2" per line.
 */
bool srhn4_tok_load(BPETokenizer *tok, const char *vocab_path) {
    if (!tok || !vocab_path) return false;

    FILE *f = fopen(vocab_path, "r");
    if (!f) {
        fprintf(stderr, "[tokenizer] Cannot open %s — using simple tokenizer\n", vocab_path);
        return false;
    }

    char line[512];
    uint32_t rank = 0;
    bool first = true;

    while (fgets(line, sizeof(line), f) && tok->n_merges < BPE_MAX_MERGES) {
        /* Strip newline */
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';

        if (first) { first = false; continue; } /* skip version comment */
        if (!line[0] || line[0] == '#') continue;

        /* Parse "left right" */
        char *sp = strchr(line, ' ');
        if (!sp) continue;
        *sp = '\0';
        char *right = sp + 1;

        BPEMerge *m = &tok->merges[tok->n_merges++];
        strncpy(m->left,   line,  MAX_TOKEN_LEN-1);
        strncpy(m->right,  right, MAX_TOKEN_LEN-1);
        snprintf(m->result, MAX_TOKEN_LEN, "%s%s", line, right);
        m->rank = rank++;
    }

    fclose(f);
    tok->loaded = tok->n_merges > 0;
    fprintf(stderr, "[tokenizer] Loaded %u BPE merges\n", tok->n_merges);

    /* Try to load vocab.json from same directory */
    char vocab_json[512];
    snprintf(vocab_json, sizeof(vocab_json), "%.*s/vocab.json",
             (int)(strrchr(vocab_path, '/') ? strrchr(vocab_path, '/') - vocab_path
                                            : (int)strlen(vocab_path)),
             vocab_path);

    FILE *vf = fopen(vocab_json, "r");
    if (vf) {
        /* Parse simple {"token": id} JSON */
        char vline[256];
        while (fgets(vline, sizeof(vline), vf)) {
            /* Look for "token": id pattern */
            char *q1 = strchr(vline, '"');
            if (!q1) continue;
            char *q2 = strchr(q1 + 1, '"');
            if (!q2) continue;
            *q2 = '\0';
            char *token = q1 + 1;
            char *colon = strchr(q2 + 1, ':');
            if (!colon) continue;
            int id = atoi(colon + 1);
            if (id >= 0 && id < BPE_MAX_VOCAB)
                vocab_insert_tok(tok, token, (uint16_t)id);
        }
        fclose(vf);
        tok->has_vocab = tok->n_vocab > 0;
        fprintf(stderr, "[tokenizer] Loaded %u vocab entries\n", tok->n_vocab);
    }

    return tok->loaded;
}

/*
 * srhn4_tok_encode — encode text to BPE token IDs.
 * Returns number of tokens produced.
 */
int srhn4_tok_encode(BPETokenizer *tok, const char *text, uint16_t *ids, int max_ids) {
    if (!tok || !text || !ids || max_ids <= 0) return 0;

    int n_ids = 0;
    const char *p = text;

    while (*p && n_ids < max_ids - 1) {
        /* Skip whitespace, record if word boundary */
        bool space_before = false;
        while (*p == ' ' || *p == '\t') { p++; space_before = true; }
        if (!*p) break;

        /* Extract one word (alpha + digits + apostrophe) */
        char word[256];
        int wi = 0;
        if (space_before && wi < 255) word[wi++] = ' ';  /* GPT-2 space prefix */

        while (*p && !isspace((unsigned char)*p) && wi < 254)
            word[wi++] = *p++;
        word[wi] = '\0';
        if (wi == 0) { p++; continue; }

        /* BPE encode this word */
        char symbols[BPE_MAX_WORD][MAX_TOKEN_LEN];
        int ns = bpe_encode_word(tok, word, symbols, BPE_MAX_WORD);

        for (int s = 0; s < ns && n_ids < max_ids - 1; s++) {
            uint16_t id = vocab_lookup(tok, symbols[s]);
            if (id == UINT16_MAX) {
                /* Unknown token → use hash-based fallback ID */
                id = (uint16_t)(fnv_tok(symbols[s]) % (BPE_MAX_VOCAB - 256)) + 256;
            }
            ids[n_ids++] = id;
        }
    }

    return n_ids;
}

/*
 * srhn4_tok_decode — decode token IDs back to text.
 */
void srhn4_tok_decode(BPETokenizer *tok, const uint16_t *ids, int n, char *out, int sz) {
    if (!tok || !ids || !out || sz <= 0) return;
    int off = 0;
    for (int i = 0; i < n && off < sz - 1; i++) {
        uint16_t id = ids[i];
        const char *token = (tok->has_vocab && id < BPE_MAX_VOCAB)
                            ? tok->id_to_token[id] : "?";
        /* Remove </w> suffix (word boundary marker) */
        char clean[MAX_TOKEN_LEN];
        strncpy(clean, token, MAX_TOKEN_LEN-1);
        char *wb = strstr(clean, "</w>");
        if (wb) { *wb = '\0'; off += snprintf(out + off, sz - off, "%s ", clean); }
        else      off += snprintf(out + off, sz - off, "%s", clean);
    }
    if (off > 0 && out[off-1] == ' ') out[off-1] = '\0'; /* trim trailing space */
}

/*
 * srhn4_tok_simple — simple word tokenizer fallback (no BPE file needed).
 * Handles punctuation, apostrophes, URLs, code tokens.
 * Returns token count.
 */
int srhn4_tok_simple(const char *text, char tokens[][MAX_WORD_LEN], int max_tok) {
    if (!text || !tokens || max_tok <= 0) return 0;

    int n = 0;
    const char *p = text;

    while (*p && n < max_tok) {
        /* Skip whitespace */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        int i = 0;
        char tok_buf[MAX_WORD_LEN];

        /* URL detection */
        if (strncmp(p, "http", 4) == 0 || strncmp(p, "www.", 4) == 0) {
            while (*p && !isspace((unsigned char)*p) && i < MAX_WORD_LEN - 1)
                tok_buf[i++] = *p++;
            tok_buf[i] = '\0';
            if (i > 0) { strncpy(tokens[n++], tok_buf, MAX_WORD_LEN-1); }
            continue;
        }

        /* Code tokens: handle :: -> && || etc */
        if (!isalnum((unsigned char)*p) && *p != '_' && *p != '\'') {
            /* Multi-char punctuation */
            char op[4] = {0};
            op[0] = *p++;
            if ((*p == '=' || *p == '>' || *p == '<' || *p == '|' ||
                 *p == '&' || *p == '+' || *p == '-') &&
                *(p-1) == op[0]) {
                op[1] = *p++;
            }
            if (op[0] && op[0] != ' ') {
                tokens[n][0] = op[0]; tokens[n][1] = op[1]; tokens[n][2] = '\0';
                if (tokens[n][0]) n++;
            }
            continue;
        }

        /* Normal alphanumeric token (includes _ for identifiers) */
        while (*p && (isalnum((unsigned char)*p) || *p == '_' || *p == '\'')
               && i < MAX_WORD_LEN - 1) {
            tok_buf[i++] = (char)tolower((unsigned char)*p++);
        }
        tok_buf[i] = '\0';

        /* Handle possessives: split "word's" → "word" "'s" */
        char *apos = strrchr(tok_buf, '\'');
        if (apos && apos > tok_buf && n < max_tok - 1) {
            *apos = '\0';
            strncpy(tokens[n++], tok_buf, MAX_WORD_LEN-1);
            if (n < max_tok) strncpy(tokens[n++], apos + 1, MAX_WORD_LEN-1);
        } else if (i > 0) {
            strncpy(tokens[n++], tok_buf, MAX_WORD_LEN-1);
        }
    }

    return n;
}
