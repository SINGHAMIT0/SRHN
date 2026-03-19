/*
 * srhn_persist.c  —  Persistence: WAL, binary save/load, JSON export
 *
 * Features:
 *   - Write-Ahead Log (WAL) for crash safety
 *   - Full binary snapshot save/load
 *   - JSON export for visualization
 *   - CRC32 checksums on all WAL entries
 */

#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── CRC32 (fast table-based) ────────────────────────────────── */
static uint32_t crc32_table[256];
static bool     crc32_ready = false;

static void crc32_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crc32_table[i] = c;
    }
    crc32_ready = true;
}

static uint32_t crc32(const void *buf, size_t len) {
    if (!crc32_ready) crc32_init();
    const uint8_t *p = (const uint8_t *)buf;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++)
        crc = crc32_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

/* ── WAL ─────────────────────────────────────────────────────── */

bool srhn4_wal_open(SRHNNetwork4 *net, const char *path) {
    WAL *wal = &net->wal;
    pthread_mutex_init(&wal->lock, NULL);
    strncpy(wal->path, path, 511);
    wal->file = fopen(path, "ab+");
    if (!wal->file) {
        fprintf(stderr, "[wal] Cannot open %s\n", path);
        return false;
    }
    /* Write header if new file */
    fseek(wal->file, 0, SEEK_END);
    long sz = ftell(wal->file);
    if (sz == 0) {
        uint32_t magic = 0x57414C33u; /* "WAL3" */
        fwrite(&magic, sizeof(magic), 1, wal->file);
        fflush(wal->file);
    }
    return true;
}

void srhn4_wal_close(SRHNNetwork4 *net) {
    WAL *wal = &net->wal;
    pthread_mutex_lock(&wal->lock);
    if (wal->file) { fflush(wal->file); fclose(wal->file); wal->file = NULL; }
    pthread_mutex_unlock(&wal->lock);
    pthread_mutex_destroy(&wal->lock);
}

bool srhn4_wal_write(SRHNNetwork4 *net, WALOpType op,
                      uint32_t a, uint32_t b, float val,
                      const void *payload, uint32_t plen) {
    WAL *wal = &net->wal;
    if (!wal->file) return false;
    if (plen > 255) plen = 255;

    WALEntry e;
    memset(&e, 0, sizeof(e));
    e.op           = op;
    e.id_a         = a;
    e.id_b         = b;
    e.value        = val;
    e.timestamp_us = srhn4_timestamp_us();
    e.payload_len  = plen;
    if (plen > 0 && payload) memcpy(e.payload, payload, plen);
    e.checksum = crc32(&e, offsetof(WALEntry, checksum));

    pthread_mutex_lock(&wal->lock);
    bool ok = fwrite(&e, sizeof(e), 1, wal->file) == 1;
    wal->n_entries++;
    wal->bytes_written += sizeof(e);

    /* Flush periodically */
    if (wal->n_entries % 32 == 0) fflush(wal->file);

    pthread_mutex_unlock(&wal->lock);
    return ok;
}

bool srhn4_wal_replay(SRHNNetwork4 *net, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    /* Skip magic header */
    uint32_t magic;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x57414C33u) {
        fclose(f); return false;
    }

    WALEntry e;
    uint32_t replayed = 0, corrupt = 0;

    while (fread(&e, sizeof(e), 1, f) == 1) {
        /* Verify checksum */
        uint32_t expected = crc32(&e, offsetof(WALEntry, checksum));
        if (e.checksum != expected) { corrupt++; continue; }

        switch (e.op) {
            case WAL_ADD_NODE:
                if (e.payload_len > 0) {
                    char label[128] = {0};
                    strncpy(label, (char *)e.payload,
                            e.payload_len < 127 ? e.payload_len : 127);
                    srhn4_add_node(net, label, (NodeType)e.id_b, LANG_ENGLISH);
                }
                break;
            case WAL_ADD_EDGE:
                srhn4_connect(net, e.id_a, e.id_b, e.value, EDGE_ASSOC);
                break;
            case WAL_UPDATE_WEIGHT:
                if (e.id_a < net->n_nodes) {
                    SRHNNode4 *node = &net->nodes[e.id_a];
                    for (uint8_t k = 0; k < node->n_neighbors; k++) {
                        if (node->neighbors[k] == e.id_b) {
                            node->edge_weights[k] = e.value; break;
                        }
                    }
                }
                break;
            case WAL_PRUNE_NODE:
                if (e.id_a < net->n_nodes)
                    net->nodes[e.id_a].type = NODE_PRUNED;
                break;
            default: break;
        }
        replayed++;
    }

    fclose(f);
    fprintf(stderr, "[wal] Replayed %u entries (%u corrupt skipped)\n",
            replayed, corrupt);
    return true;
}

/* ── Binary snapshot save ────────────────────────────────────── */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_nodes;
    uint32_t n_edges;
    uint32_t n_lex;
    uint32_t n_physics;
    uint64_t query_counter;
    float    global_time;
    uint8_t  pad[32];
} SaveHeader3;

bool srhn4_save(SRHNNetwork4 *net, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { net->last_error = SRHN_ERR_IO; return false; }

    SaveHeader3 hdr = {
        SRHN4_MAGIC, 3,
        net->n_nodes, net->n_edges,
        net->lexicon.n_entries, net->n_physics_laws,
        net->query_counter, net->global_time
    };
    memset(hdr.pad, 0, sizeof(hdr.pad));
    fwrite(&hdr, sizeof(hdr), 1, f);

    /* Write nodes (fixed-size part + variable neighbor arrays) */
    for (uint32_t i = 0; i < net->n_nodes; i++) {
        SRHNNode4 *n = &net->nodes[i];
        fwrite(n, offsetof(SRHNNode4, neighbors), 1, f);
        fwrite(&n->n_neighbors, 1, 1, f);
        if (n->n_neighbors > 0) {
            fwrite(n->neighbors,    sizeof(uint32_t), n->n_neighbors, f);
            fwrite(n->edge_weights, sizeof(float),    n->n_neighbors, f);
            /* Edge types (new in v3) */
            if (n->edge_types)
                fwrite(n->edge_types, sizeof(EdgeType), n->n_neighbors, f);
            else {
                EdgeType dummy = EDGE_ASSOC;
                for (uint8_t k = 0; k < n->n_neighbors; k++)
                    fwrite(&dummy, sizeof(EdgeType), 1, f);
            }
        }
    }

    /* Write edges */
    for (uint32_t i = 0; i < net->n_edges; i++) {
        SRHNEdge4 *e = &net->edges[i];
        fwrite(e, offsetof(SRHNEdge4, nodes), 1, f);
        fwrite(&e->n_nodes, 1, 1, f);
        if (e->n_nodes > 0) fwrite(e->nodes, sizeof(uint32_t), e->n_nodes, f);
    }

    /* Lexicon */
    fwrite(&net->lexicon.n_entries, sizeof(uint32_t), 1, f);
    fwrite(net->lexicon.entries, sizeof(LexEntry), net->lexicon.n_entries, f);

    /* Physics */
    fwrite(&net->n_physics_laws, sizeof(uint32_t), 1, f);
    fwrite(net->physics_laws, sizeof(PhysicsLaw), net->n_physics_laws, f);

    /* Stats + calibration */
    fwrite(&net->stats, sizeof(SRHNStats4),  1, f);
    fwrite(&net->calib, sizeof(CalibTable),  1, f);

    fclose(f);
    fprintf(stderr, "[persist] Saved %u nodes, %u edges to %s\n",
            net->n_nodes, net->n_edges, path);

    /* Truncate WAL after successful snapshot */
    if (net->wal.file) {
        fclose(net->wal.file);
        net->wal.file = fopen(net->wal.path, "wb");
        if (net->wal.file) {
            uint32_t magic = 0x57414C33u;
            fwrite(&magic, sizeof(magic), 1, net->wal.file);
            fflush(net->wal.file);
        }
        net->wal.n_entries   = 0;
        net->wal.bytes_written = 0;
    }

    return true;
}

/* ── Binary snapshot load ────────────────────────────────────── */

SRHNNetwork4 *srhn4_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    SaveHeader3 hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.magic != SRHN4_MAGIC) {
        fprintf(stderr, "[persist] Bad magic in %s\n", path);
        fclose(f); return NULL;
    }

    SRHNNetwork4 *net = srhn4_create();
    if (!net) { fclose(f); return NULL; }

    /* Ensure capacity */
    while (net->node_cap < hdr.n_nodes) {
        uint32_t nc = net->node_cap * 2;
        SRHNNode4 *p = (SRHNNode4 *)realloc(net->nodes, nc * sizeof(SRHNNode4));
        if (!p) { srhn4_destroy(net); fclose(f); return NULL; }
        memset(p + net->node_cap, 0, (nc - net->node_cap) * sizeof(SRHNNode4));
        net->nodes = p; net->node_cap = nc;
    }
    while (net->edge_cap < hdr.n_edges) {
        uint32_t nc = net->edge_cap * 2;
        SRHNEdge4 *p = (SRHNEdge4 *)realloc(net->edges, nc * sizeof(SRHNEdge4));
        if (!p) { srhn4_destroy(net); fclose(f); return NULL; }
        memset(p + net->edge_cap, 0, (nc - net->edge_cap) * sizeof(SRHNEdge4));
        net->edges = p; net->edge_cap = nc;
    }

    /* Load nodes */
    for (uint32_t i = 0; i < hdr.n_nodes; i++) {
        SRHNNode4 *n = &net->nodes[i];
        free(n->neighbors); free(n->edge_weights); free(n->edge_types);

        if (fread(n, offsetof(SRHNNode4, neighbors), 1, f) != 1) goto load_err;

        uint8_t nc; fread(&nc, 1, 1, f);
        n->n_neighbors  = nc;
        n->neighbor_cap = nc > 8 ? nc : 8;
        n->neighbors    = (uint32_t *)malloc(n->neighbor_cap * sizeof(uint32_t));
        n->edge_weights = (float    *)malloc(n->neighbor_cap * sizeof(float));
        n->edge_types   = (EdgeType *)malloc(n->neighbor_cap * sizeof(EdgeType));

        if (!n->neighbors || !n->edge_weights || !n->edge_types) goto load_err;

        /* Give extra capacity headroom so first connect() never needs immediate realloc */
        n->neighbor_cap = (uint16_t)(nc < 8 ? 16 : (nc < 120 ? nc + 16 : 128));
        /* Re-allocate with new cap */
        free(n->neighbors);    n->neighbors    = (uint32_t *)malloc(n->neighbor_cap * sizeof(uint32_t));
        free(n->edge_weights); n->edge_weights = (float    *)malloc(n->neighbor_cap * sizeof(float));
        free(n->edge_types);   n->edge_types   = (EdgeType *)malloc(n->neighbor_cap * sizeof(EdgeType));
        if (!n->neighbors || !n->edge_weights || !n->edge_types) goto load_err;
        n->edge_timestamps = (uint64_t *)calloc(n->neighbor_cap, sizeof(uint64_t));
        if (nc > 0) {
            fread(n->neighbors,    sizeof(uint32_t), nc, f);
            fread(n->edge_weights, sizeof(float),    nc, f);
            fread(n->edge_types,   sizeof(EdgeType), nc, f);
        }
        /* Validate: cap must be >= n_neighbors to avoid immediate corruption */
        if (n->neighbor_cap < n->n_neighbors) n->neighbor_cap = (uint16_t)(n->n_neighbors + 8);
    }
    net->n_nodes = hdr.n_nodes;

    /* Load edges */
    for (uint32_t i = 0; i < hdr.n_edges; i++) {
        SRHNEdge4 *e = &net->edges[i];
        free(e->nodes);
        if (fread(e, offsetof(SRHNEdge4, nodes), 1, f) != 1) goto load_err;
        uint8_t nc; fread(&nc, 1, 1, f);
        e->n_nodes = nc;
        e->nodes = (uint32_t *)malloc((nc > 0 ? nc : 1) * sizeof(uint32_t));
        if (!e->nodes) goto load_err;
        if (nc > 0) fread(e->nodes, sizeof(uint32_t), nc, f);
    }
    net->n_edges = hdr.n_edges;

    /* Lexicon */
    uint32_t lex_n; fread(&lex_n, sizeof(uint32_t), 1, f);
    if (lex_n > 0) {
        while (net->lexicon.cap < lex_n) {
            net->lexicon.cap *= 2;
            net->lexicon.entries = (LexEntry *)realloc(net->lexicon.entries,
                                    net->lexicon.cap * sizeof(LexEntry));
        }
        fread(net->lexicon.entries, sizeof(LexEntry), lex_n, f);
        net->lexicon.n_entries = lex_n;
    }

    /* Physics */
    uint32_t pl_n; fread(&pl_n, sizeof(uint32_t), 1, f);
    if (pl_n > 0 && pl_n <= 512) {
        fread(net->physics_laws, sizeof(PhysicsLaw), pl_n, f);
        net->n_physics_laws = pl_n;
    }

    /* Stats + calibration */
    fread(&net->stats, sizeof(SRHNStats4), 1, f);
    fread(&net->calib, sizeof(CalibTable), 1, f);

    net->query_counter = hdr.query_counter;
    net->global_time   = hdr.global_time;

    fclose(f);

    /* Rebuild HNSW index */
    fprintf(stderr, "[persist] Rebuilding HNSW index for %u nodes...\n", net->n_nodes);
    for (uint32_t i = 0; i < net->n_nodes; i++) {
        if (net->nodes[i].type != NODE_PRUNED)
            srhn4_hnsw_insert(&net->hnsw, i, &net->nodes[i].sig, net);
    }

    /* Try to replay WAL */
    char wal_path[512];
    snprintf(wal_path, sizeof(wal_path), "%s.wal", path);
    srhn4_wal_replay(net, wal_path);

    fprintf(stderr, "[persist] Loaded %u nodes, %u edges from %s\n",
            net->n_nodes, net->n_edges, path);
    return net;

load_err:
    fprintf(stderr, "[persist] Error reading %s\n", path);
    fclose(f);
    srhn4_destroy(net);
    return NULL;
}

/* ── JSON export ─────────────────────────────────────────────── */

static void json_escape(const char *in, char *out, int sz) {
    int oi = 0;
    for (int i = 0; in[i] && oi < sz - 2; i++) {
        char c = in[i];
        if (c == '"' || c == '\\') { out[oi++] = '\\'; }
        else if (c == '\n') { out[oi++] = '\\'; out[oi++] = 'n'; continue; }
        else if (c == '\t') { out[oi++] = '\\'; out[oi++] = 't'; continue; }
        if (oi < sz - 1) out[oi++] = c;
    }
    out[oi] = '\0';
}

void srhn4_export_json(SRHNNetwork4 *net, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return;

    SRHNStats4 s = srhn4_get_stats(net);

    fprintf(f, "{\n"
               "  \"version\":\"%s\",\n"
               "  \"stats\":{\n"
               "    \"total_nodes\":%u,\n"
               "    \"active_nodes\":%u,\n"
               "    \"total_edges\":%u,\n"
               "    \"auto_nodes\":%u,\n"
               "    \"avg_entropy\":%.4f,\n"
               "    \"total_queries\":%llu,\n"
               "    \"avg_latency_ms\":%.3f,\n"
               "    \"feedback_score\":%.3f,\n"
               "    \"avg_pagerank\":%.6f,\n"
               "    \"ae_runs\":%u,\n"
               "    \"ae_nodes_promoted\":%u,\n"
               "    \"spectral_runs\":%u,\n"
               "    \"avg_pacbayes_bound\":%.6f,\n"
               "    \"avg_msg_entropy\":%.4f,\n"
               "    \"fast_ring_count\":%u,\n"
               "    \"hnsw_hits\":%u\n"
               "  },\n",
               SRHN4_VERSION,
               s.total_nodes, s.active_nodes, s.total_edges, s.auto_nodes,
               s.avg_entropy, (unsigned long long)s.total_queries,
               s.avg_latency_ms, s.feedback_score, s.avg_pagerank,
               s.ae_runs, s.ae_nodes_promoted, s.spectral_runs,
               s.avg_pacbayes_bound, s.avg_msg_entropy,
               s.fast_ring_count, s.hnsw_hits);

    /* Nodes */
    fprintf(f, "  \"nodes\":[\n");
    bool first = true;
    char esc[256];
    for (uint32_t i = 0; i < net->n_nodes; i++) {
        SRHNNode4 *n = &net->nodes[i];
        if (n->type == NODE_PRUNED) continue;
        if (!first) fprintf(f, ",\n");
        first = false;
        json_escape(n->label, esc, 255);
        /* Build compact sig embedding (32 dims) for visualizer */
        char vbuf[300]; int vi=0;
        vi += snprintf(vbuf+vi, 299-vi, "[");
        for(int _d=0; _d<32 && _d<EMBED_DIM; _d++)
            vi += snprintf(vbuf+vi, 299-vi, "%s%.4f", _d?",":"", n->sig.vec[_d]);
        snprintf(vbuf+vi, 299-vi, "]");
        char fesc2[128]="", cesc2[32]="";
        if (n->formula[0])   json_escape(n->formula,   fesc2, 127);
        if (n->code_lang[0]) json_escape(n->code_lang, cesc2, 31);
        fprintf(f,
            "    {\"id\":%u,\"label\":\"%s\","
            "\"type\":%d,\"lang\":%d,"
            "\"activation\":%.3f,\"entropy_score\":%.3f,"
            "\"importance\":%.4f,"
            "\"in_fast_ring\":%s,\"fast\":%s,"
            "\"usage_count\":%.1f,\"auto_created\":%s,"
            "\"hnsw_layer\":%d,\"n_neighbors\":%d,"
            "\"ae_recon_error\":%.4f,"
            "\"formula\":\"%s\",\"code_lang\":\"%s\","
            "\"sig\":{\"vec\":%s,"
            "\"view_weights\":[%.3f,%.3f,%.3f]}}",
            n->id, esc,
            (int)n->type, (int)n->lang,
            n->activation, n->entropy_score,
            n->importance,
            n->in_fast_ring?"true":"false",
            n->in_fast_ring?"true":"false",
            n->usage_count, n->auto_created?"true":"false",
            (int)n->hnsw_layer, (int)n->n_neighbors,
            n->ae_recon_error,
            fesc2, cesc2, vbuf,
            n->sig.view_weights[0],
            n->sig.view_weights[1],
            n->sig.view_weights[2]);
    }
    fprintf(f, "\n  ],\n");

    /* Edges — neighbor-pair edges */
    fprintf(f, "  \"edges\":[\n");
    first = true;
    static const char *etype_names[] = {
        "assoc","causal","synonym","antonym","instance","part_of",
        "causes","inhibitory","temporal","spatial","code","visual",
        "contradicts","supports","unknown"
    };
    for (uint32_t i = 0; i < net->n_nodes; i++) {
        SRHNNode4 *n = &net->nodes[i];
        if (n->type == NODE_PRUNED) continue;
        for (uint8_t k = 0; k < n->n_neighbors; k++) {
            uint32_t j = n->neighbors[k];
            if (j <= i) continue; /* avoid duplicates */
            if (j >= net->n_nodes) continue;
            if (!first) fprintf(f, ",\n"); first = false;
            EdgeType et = n->edge_types ? n->edge_types[k] : EDGE_ASSOC;
            int eti = (int)et < 15 ? (int)et : 14;
            fprintf(f,
                "    {\"source\":%u,\"target\":%u,\"weight\":%.3f,"
                "\"type\":%d,\"type_name\":\"%s\",\"causal\":%s}",
                i, j, n->edge_weights[k],
                eti, etype_names[eti],
                (eti==1||eti==6)?"true":"false");
        }
    }
    /* Hyperedges */
    for (uint32_t e = 0; e < net->n_edges; e++) {
        SRHNEdge4 *edge = &net->edges[e];
        if (edge->n_nodes < 2) continue;
        for (uint8_t i = 0; i < edge->n_nodes - 1; i++) {
            for (uint8_t j = i + 1; j < edge->n_nodes; j++) {
                if (!first) fprintf(f, ",\n"); first = false;
                json_escape(edge->relation, esc, 255);
                fprintf(f,
                    "    {\"source\":%u,\"target\":%u,\"weight\":%.3f,"
                    "\"type\":\"hyper\",\"relation\":\"%s\","
                    "\"causal\":%s,\"activation\":%.3f}",
                    edge->nodes[i], edge->nodes[j], edge->weight,
                    esc, edge->is_causal ? "true" : "false",
                    edge->activation);
            }
        }
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    fprintf(stderr, "[persist] Exported JSON to %s\n", path);
}
