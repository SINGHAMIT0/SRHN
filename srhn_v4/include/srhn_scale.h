/*
 * srhn_scale.h  —  SRHN v4 Scalability & Safety Layer
 *
 * Provides:
 *  1. Memory-mapped slab allocator for node/edge arrays (TB-scale)
 *  2. LRU node cache (hot set stays in RAM, cold nodes paged to disk)
 *  3. Streaming document ingestor (line-at-a-time, bounded RAM)
 *  4. Sharded HNSW index for parallel insertion
 *  5. WAL rotation (size-bounded, automatic compaction)
 *  6. Safe string / integer helpers (overflow, bounds, encoding)
 *  7. Edge-case guards injected at API surface
 *  8. Memory pressure watchdog
 */
#pragma once
#ifndef SRHN_SCALE_H
#define SRHN_SCALE_H

#define _POSIX_C_SOURCE 200809L
#include "srhn_v4.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>

/* ── Configuration ──────────────────────────────────────────── */

/* Maximum RAM to use for node storage before spilling to mmap */
#define SCALE_MAX_RAM_NODES_MB    2048    /* 2GB of node objects */
#define SCALE_MAX_RAM_EDGES_MB    1024

/* LRU hot cache size (nodes kept always-warm) */
#define SCALE_LRU_CAPACITY        65536   /* 64K hot nodes */
#define SCALE_LRU_EVICT_BATCH     1024    /* evict this many at once */

/* Streaming ingest chunk size (bytes) */
#define SCALE_INGEST_CHUNK        65536   /* 64KB per read */
#define SCALE_INGEST_LINE_MAX     4096    /* max line length */

/* WAL rotation */
#define SCALE_WAL_MAX_BYTES       (512ULL * 1024 * 1024)  /* 512MB */
#define SCALE_WAL_COMPACT_EVERY   1000000  /* ops between compactions */

/* BFS visited array: use bitset for large graphs */
#define SCALE_BITSET_THRESHOLD    100000  /* nodes above this use bitset */

/* Memory pressure */
#define SCALE_PRESSURE_HIGH_PCT   85      /* % of system RAM → emergency prune */
#define SCALE_PRESSURE_WARN_PCT   70

/* ── Bitset visited array for large-graph BFS ────────────────── */
typedef struct {
    uint64_t *bits;
    uint32_t  n_words;
    uint32_t  n_nodes;
} Bitset;

static inline Bitset *bitset_create(uint32_t n) {
    Bitset *b = (Bitset *)calloc(1, sizeof(Bitset));
    if (!b) return NULL;
    b->n_nodes = n;
    b->n_words = (n + 63) / 64;
    b->bits    = (uint64_t *)calloc(b->n_words, sizeof(uint64_t));
    if (!b->bits) { free(b); return NULL; }
    return b;
}
static inline void bitset_free(Bitset *b) { if(b){free(b->bits);free(b);} }
static inline bool bitset_test(Bitset *b, uint32_t i) {
    if (i >= b->n_nodes) return false;
    return (b->bits[i/64] >> (i%64)) & 1;
}
static inline void bitset_set(Bitset *b, uint32_t i) {
    if (i < b->n_nodes) b->bits[i/64] |= (1ULL << (i%64));
}
static inline void bitset_clear_all(Bitset *b) {
    if (b) memset(b->bits, 0, b->n_words * sizeof(uint64_t));
}

/* ── LRU cache entry ─────────────────────────────────────────── */
typedef struct LRUEntry {
    uint32_t         node_id;
    uint64_t         last_access_us;
    struct LRUEntry *prev, *next;
} LRUEntry;

typedef struct {
    LRUEntry  *head, *tail;       /* MRU … LRU doubly-linked list */
    LRUEntry **table;             /* hash table: node_id → entry  */
    uint32_t   capacity;
    uint32_t   size;
    uint64_t   hits, misses, evictions;
    pthread_mutex_t lock;
} LRUCache;

/* ── Memory-mapped slab ──────────────────────────────────────── */
typedef struct {
    void    *ptr;           /* mmap base */
    size_t   mapped_bytes;
    int      fd;
    char     path[512];
    bool     writable;
    bool     anonymous;     /* anonymous mmap (no file) */
} MmapSlab;

/* ── Scale context attached to network ───────────────────────── */
typedef struct {
    /* LRU cache */
    LRUCache   lru;
    bool       lru_enabled;

    /* Memory-mapped node/edge backing store */
    MmapSlab   node_slab;
    MmapSlab   edge_slab;
    bool       mmap_enabled;

    /* Streaming state */
    bool       streaming_ingest;
    uint64_t   bytes_ingested;
    uint32_t   docs_ingested;
    uint32_t   nodes_from_ingest;

    /* WAL rotation */
    uint64_t   wal_bytes_written;
    uint32_t   wal_generation;
    char       wal_base_path[512];

    /* Bitset pool for BFS (reused across queries) */
    Bitset    *bfs_bitset;
    pthread_mutex_t bfs_lock;

    /* Memory pressure */
    uint64_t   system_ram_bytes;
    float      last_pressure_pct;
    uint64_t   last_pressure_check_us;

    /* Stats */
    uint64_t   emergency_prunes;
    uint64_t   wal_rotations;
    uint64_t   mmap_page_faults_estimated;
} ScaleContext;

/* ── Safe string helpers ─────────────────────────────────────── */
#define SRHN_SAFE_COPY(dst, src, dstsz) do { \
    if ((src) && (dstsz) > 0) { \
        strncpy((dst), (src), (dstsz)-1); \
        (dst)[(dstsz)-1] = '\0'; \
    } \
} while(0)

/* Safe uint32 addition — returns false on overflow */
static inline bool safe_add_u32(uint32_t a, uint32_t b, uint32_t *out) {
    if (b > UINT32_MAX - a) return false;
    *out = a + b; return true;
}

/* Safe size_t multiply — returns false on overflow */
static inline bool safe_mul_sz(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b; return true;
}

/* Validate UTF-8 (returns false if invalid) */
bool srhn_scale_valid_utf8(const char *s, size_t len);

/* Sanitise label: strip control chars, limit length, ensure UTF-8 */
void srhn_scale_sanitise_label(char *dst, const char *src, size_t dst_sz);

/* ── Public API ──────────────────────────────────────────────── */

/* Init / teardown */
bool srhn_scale_init   (SRHNNetwork4 *net, const char *data_dir);
void srhn_scale_destroy(SRHNNetwork4 *net);

/* LRU */
bool srhn_scale_lru_init    (LRUCache *cache, uint32_t capacity);
void srhn_scale_lru_free    (LRUCache *cache);
void srhn_scale_lru_touch   (LRUCache *cache, uint32_t node_id);
bool srhn_scale_lru_contains(LRUCache *cache, uint32_t node_id);
void srhn_scale_lru_evict   (LRUCache *cache, uint32_t n);

/* BFS bitset (thread-safe via internal lock) */
Bitset *srhn_scale_bfs_bitset_acquire(SRHNNetwork4 *net);
void    srhn_scale_bfs_bitset_release(SRHNNetwork4 *net, Bitset *b);

/* Streaming ingest */
int  srhn_scale_ingest_stream(SRHNNetwork4 *net, FILE *f, uint32_t source_id,
                               void (*progress_cb)(uint64_t bytes, uint32_t nodes, void *ud),
                               void *userdata);
int  srhn_scale_ingest_path  (SRHNNetwork4 *net, const char *path, uint32_t source_id);

/* WAL rotation */
bool srhn_scale_wal_maybe_rotate(SRHNNetwork4 *net);
bool srhn_scale_wal_compact     (SRHNNetwork4 *net);

/* Memory pressure */
float srhn_scale_memory_pressure(SRHNNetwork4 *net);
void  srhn_scale_emergency_prune(SRHNNetwork4 *net, float target_pct);

/* Mmap backing store */
bool srhn_scale_mmap_nodes(SRHNNetwork4 *net, const char *path, uint32_t max_nodes);
bool srhn_scale_mmap_edges(SRHNNetwork4 *net, const char *path, uint32_t max_edges);

/* Scale context accessor (embedded in network via void* user_data) */
ScaleContext *srhn_scale_ctx(SRHNNetwork4 *net);

#endif /* SRHN_SCALE_H */
