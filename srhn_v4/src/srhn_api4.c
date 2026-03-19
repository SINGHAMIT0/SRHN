/*
 * srhn_api.c  —  Minimal REST API server for SRHN v3
 *
 * Endpoints:
 *   POST /query        { "q": "..." }  → { "response": "...", "confidence": 0.87, ... }
 *   POST /feedback     { "query_id": N, "reward": 0.8 }
 *   GET  /stats        → JSON stats
 *   GET  /graph        → full JSON graph export
 *   POST /ingest       { "text": "...", "source_id": N }
 *   GET  /health       → { "status": "ok" }
 *
 * Uses blocking POSIX sockets — one thread per connection (adequate for dev/research use).
 * For production, swap the accept loop for libevent or io_uring.
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <signal.h>
#include <ctype.h>
#include <fcntl.h>
#include <sys/time.h>

/* ── Server state ─────────────────────────────────────────────── */
typedef struct SRHNServer4 SRHNServer;
struct SRHNServer4 {
    SRHNNetwork4 *net;
    int           listen_fd;
    int           port;
    volatile bool running;
    pthread_t     accept_thread;
};

/* ── Minimal JSON helpers ────────────────────────────────────── */

static void json_str(char *buf, int sz, const char *key, const char *val) {
    char esc[RESP_BUF_SIZE + 64];
    int ei = 0;
    for (int i = 0; val[i] && ei < (int)sizeof(esc) - 3; i++) {
        if (val[i] == '"')       { esc[ei++] = '\\'; esc[ei++] = '"'; }
        else if (val[i] == '\\') { esc[ei++] = '\\'; esc[ei++] = '\\'; }
        else if (val[i] == '\n') { esc[ei++] = '\\'; esc[ei++] = 'n'; }
        else if (val[i] == '\r') { esc[ei++] = '\\'; esc[ei++] = 'r'; }
        else if (val[i] == '\t') { esc[ei++] = '\\'; esc[ei++] = 't'; }
        else esc[ei++] = val[i];
    }
    esc[ei] = '\0';
    snprintf(buf, sz, "\"%s\":\"%s\"", key, esc);
}

/* Extract string value from simple JSON: {"key":"value"} */
static bool json_get_str(const char *json, const char *key,
                          char *out, int out_sz) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return false;
    p += strlen(pat);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    if (*p == '"') {
        p++;
        int i = 0;
        while (*p && *p != '"' && i < out_sz - 1) {
            if (*p == '\\' && *(p+1)) { p++; }
            out[i++] = *p++;
        }
        out[i] = '\0';
        return i > 0;
    }
    return false;
}

static double json_get_num(const char *json, const char *key) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0.0;
    p += strlen(pat);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    return atof(p);
}

/* ── HTTP helpers ─────────────────────────────────────────────── */

static void send_response(int fd, int status, const char *content_type,
                           const char *body, int body_len) {
    char hdr[512];
    const char *status_str = (status == 200) ? "OK" :
                              (status == 400) ? "Bad Request" :
                              (status == 404) ? "Not Found" :
                              (status == 405) ? "Method Not Allowed" :
                                                "Internal Server Error";
    int hlen = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization, X-Requested-With\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_str, content_type, body_len);
    send(fd, hdr, hlen, MSG_NOSIGNAL);
    if (body && body_len > 0) send(fd, body, body_len, MSG_NOSIGNAL);
}

static void send_json(int fd, const char *json) {
    send_response(fd, 200, "application/json", json, (int)strlen(json));
}

static void send_error(int fd, int status, const char *msg) {
    char buf[256];
    snprintf(buf, sizeof(buf), "{\"error\":\"%s\"}", msg);
    send_response(fd, status, "application/json", buf, (int)strlen(buf));
}

/* ── Request handlers ─────────────────────────────────────────── */

static char *esc_str(char *dst, size_t dsz, const char *src) {
    size_t o = 0;
    if (!src) { if (dsz > 0) dst[0] = 0; return dst; }
    for (size_t i = 0; src[i] && o < dsz - 3; i++) {
        unsigned char c = (unsigned char)src[i];
        if      (c == '"')  { dst[o++] = '\\'; dst[o++] = '"'; }
        else if (c == '\\') { dst[o++] = '\\'; dst[o++] = '\\'; }
        else if (c == '\n') { dst[o++] = '\\'; dst[o++] = 'n'; }
        else if (c == '\r') { dst[o++] = '\\'; dst[o++] = 'r'; }
        else if (c == '\t') { dst[o++] = '\\'; dst[o++] = 't'; }
        else if (c >= 0x20)  { dst[o++] = (char)c; }
    }
    dst[o] = 0;
    return dst;
}

static void handle_query(int fd, SRHNNetwork4 *net,
                          const char *body, int blen) {
    (void)blen;
    char q[1024] = "";
    if (!json_get_str(body, "q", q, sizeof(q)) &&
        !json_get_str(body, "query", q, sizeof(q)))
        { send_error(fd, 400, "missing q"); return; }
    if (!q[0]) { send_error(fd, 400, "empty q"); return; }

    /* Execute query — allocate result on heap (SRHNResult4 is ~18KB) */
    SRHNResult4 *res = (SRHNResult4 *)calloc(1, sizeof(SRHNResult4));
    if (!res) { send_error(fd, 500, "oom"); return; }
    *res = srhn4_query(net, q);

    /* Build JSON response on heap */
    char *resp   = (char *)calloc(1, 131072);
    char *esc_r  = (char *)calloc(1, RESP_BUF_SIZE + 64);
    if (!resp || !esc_r) {
        free(resp); free(esc_r); free(res);
        send_error(fd, 500, "oom"); return;
    }

    esc_str(esc_r, RESP_BUF_SIZE + 63, res->response);

    int  off = 0, cap = 131070;
    char tmp[512];

#define W(...) do { int _n=snprintf(resp+off,cap-off,__VA_ARGS__); if(_n>0&&_n<cap-off)off+=_n; } while(0)

    W("{\"response\":\"%s\",", esc_r);
    W("\"confidence\":%.4f,\"confidence_lb\":%.4f,\"confidence_ub\":%.4f,",
        res->confidence, res->confidence_lb, res->confidence_ub);
    W("\"n_activated\":%u,\"n_chains\":%u,\"n_contradictions\":%u,",
        res->n_activated, res->n_chains, res->n_contradictions);
    W("\"best_resonance\":%.4f,\"latency_ms\":%.3f,\"llm_latency_ms\":%.3f,",
        res->best_resonance, res->latency_ms, res->llm_latency_ms);
    W("\"used_llm\":%s,\"response_grounded\":%s,",
        res->used_llm?"true":"false", res->response_grounded?"true":"false");
    W("\"msg_entropy\":%.4f,\"max_attn_score\":%.4f,",
        res->msg_entropy, res->max_attn_score);
    W("\"seeds_used\":%u,\"hops_taken\":%u,",
        res->seeds_used, res->hops_taken);

    W("\"top_concepts\":[");
    for (uint8_t i = 0; i < res->n_top_concepts && i < 8; i++) {
        esc_str(tmp, sizeof(tmp), res->top_concepts[i]);
        W("%s\"%s\"", i?",":"", tmp);
    }
    W("],");

    W("\"chains\":[");
    for (uint8_t c = 0; c < res->n_chains && c < 3; c++) {
        ReasoningChain *ch = &res->chains[c];
        W("%s{\"strength\":%.3f,\"length\":%u,\"causal\":%s,\"labels\":[",
            c?",":"", ch->total_strength, ch->length,
            ch->is_causal?"true":"false");
        for (uint8_t l = 0; l < ch->length && l < MAX_CHAIN_LEN; l++) {
            esc_str(tmp, sizeof(tmp), ch->labels[l]);
            W("%s\"%s\"", l?",":"", tmp);
        }
        W("]}");
    }
    W("],");

    W("\"contradictions\":[");
    for (uint8_t c = 0; c < res->n_contradictions; c++) {
        char la[256], lb[256];
        esc_str(la, sizeof(la), res->contradictions[c].label_a);
        esc_str(lb, sizeof(lb), res->contradictions[c].label_b);
        W("%s{\"a\":\"%s\",\"b\":\"%s\",\"conf\":%.3f}",
            c?",":"", la, lb, res->contradictions[c].confidence);
    }
    W("]}");
#undef W

    send_json(fd, resp);
    free(esc_r); free(resp); free(res);
}

static void handle_feedback(int fd, SRHNNetwork4 *net,
                              const char *body, int blen) {
    (void)blen;
    double reward = json_get_num(body, "reward");
    double qid    = json_get_num(body, "query_id");
    if (reward == 0.0 && !strstr(body, "reward")) {
        send_error(fd, 400, "Missing 'reward' field"); return;
    }
    srhn4_feedback(net, (uint32_t)qid, (float)reward);
    send_json(fd, "{\"status\":\"ok\"}");
}

static void handle_stats(int fd, SRHNNetwork4 *net) {
    SRHNStats4 s = srhn4_get_stats(net);
    char *buf = (char *)calloc(1, 4096);
    if (!buf) { send_error(fd, 500, "oom"); return; }
    snprintf(buf, 4095,
        "{"
        "\"version\":\"%s\","
        "\"total_nodes\":%u,"
        "\"active_nodes\":%u,"
        "\"pruned_nodes\":%u,"
        "\"total_edges\":%u,"
        "\"fast_ring_count\":%u,"
        "\"auto_nodes\":%u,"
        "\"avg_entropy\":%.4f,"
        "\"avg_pagerank\":%.6f,"
        "\"total_queries\":%llu,"
        "\"avg_latency_ms\":%.3f,"
        "\"llm_calls\":%u,"
        "\"hnsw_hits\":%u,"
        "\"hnsw_misses\":%u,"
        "\"feedback_score\":%.4f,"
        "\"hebbian_updates\":%llu,"
        "\"selfgrow_count\":%u,"
        "\"physics_laws\":%u,"
        "\"ae_runs\":%u,"
        "\"ae_nodes_promoted\":%u,"
        "\"ae_nodes_pruned\":%u,"
        "\"vae_candidates_generated\":%u,"
        "\"vae_candidates_accepted\":%u,"
        "\"spectral_runs\":%u,"
        "\"avg_pacbayes_bound\":%.6f,"
        "\"avg_msg_entropy\":%.4f"
        "}",
        SRHN4_VERSION,
        s.total_nodes, s.active_nodes, s.pruned_nodes,
        s.total_edges, s.fast_ring_count, s.auto_nodes,
        s.avg_entropy, s.avg_pagerank,
        (unsigned long long)s.total_queries,
        s.avg_latency_ms, s.llm_calls,
        s.hnsw_hits, s.hnsw_misses,
        s.feedback_score,
        (unsigned long long)s.hebbian_updates,
        net->selfgrow_count,
        net->n_physics_laws,
        s.ae_runs, s.ae_nodes_promoted, s.ae_nodes_pruned,
        s.vae_candidates_generated, s.vae_candidates_accepted,
        s.spectral_runs,
        s.avg_pacbayes_bound,
        s.avg_msg_entropy);
    send_json(fd, buf);
    free(buf);
}

static void handle_ingest(int fd, SRHNNetwork4 *net,
                           const char *body, int blen) {
    (void)blen;
    char text[8192] = "";
    if (!json_get_str(body, "text", text, sizeof(text))) {
        send_error(fd, 400, "Missing 'text' field"); return;
    }
    double sid = json_get_num(body, "source_id");
    srhn4_ingest_document(net, text, (uint32_t)sid);
    char resp[128];
    snprintf(resp, sizeof(resp), "{\"status\":\"ok\",\"nodes\":%u}", net->n_nodes);
    send_json(fd, resp);
}

static void handle_graph(int fd, SRHNNetwork4 *net) {
    /* Export to temp file then read back */
    const char *tmp = "/tmp/srhn4_graph_export.json";
    srhn4_export_json(net, tmp);
    FILE *f = fopen(tmp, "r");
    if (!f) { send_error(fd, 500, "Export failed"); return; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char *buf = (char *)malloc(sz + 1);
    if (!buf) { fclose(f); send_error(fd, 500, "OOM"); return; }
    fread(buf, 1, sz, f); buf[sz] = '\0'; fclose(f);
    send_response(fd, 200, "application/json", buf, (int)sz);
    free(buf);
}

/* ── Connection handler ──────────────────────────────────────── */

typedef struct { int fd; SRHNNetwork4 *net; } ConnArgs;

static void *handle_conn(void *arg) {
    ConnArgs *ca  = (ConnArgs *)arg;
    int       fd  = ca->fd;
    SRHNNetwork4 *net = ca->net;
    free(ca);

    char *buf = (char *)malloc(API_READ_BUF);
    if (!buf) { close(fd); return NULL; }

    int received = 0;
    /* Read until double CRLF (end of headers) */
    while (received < API_READ_BUF - 1) {
        int n = (int)recv(fd, buf + received, API_READ_BUF - 1 - received, 0);
        if (n <= 0) break;
        received += n;
        buf[received] = '\0';
        if (strstr(buf, "\r\n\r\n")) break;
    }
    buf[received] = '\0';

    /* Parse method and path */
    char method[8] = "", path[256] = "";
    sscanf(buf, "%7s %255s", method, path);

    /* Find body (after double CRLF) */
    const char *body_start = strstr(buf, "\r\n\r\n");
    const char *body = body_start ? body_start + 4 : "";
    int body_len = received - (int)(body - buf);

    /* Content-Length — read more if needed */
    const char *cl_hdr = strcasestr(buf, "Content-Length:");
    if (cl_hdr) {
        int cl = atoi(cl_hdr + 15);
        while (body_len < cl && received < API_READ_BUF - 1) {
            int n = (int)recv(fd, buf + received, cl - body_len, 0);
            if (n <= 0) break;
            received += n; body_len += n;
        }
        buf[received] = '\0';
        body = body_start ? body_start + 4 : "";
        body_len = received - (int)(body - buf);
    }

    /* Route */
    if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
        send_json(fd, "{\"status\":\"ok\",\"version\":\"" SRHN4_VERSION "\"}");
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/stats") == 0) {
        handle_stats(fd, net);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/graph") == 0) {
        handle_graph(fd, net);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/query") == 0) {
        handle_query(fd, net, body, body_len);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/feedback") == 0) {
        handle_feedback(fd, net, body, body_len);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/ingest") == 0) {
        handle_ingest(fd, net, body, body_len);
    } else if (strcmp(method, "OPTIONS") == 0) {
        /* CORS preflight — 204 No Content with all CORS headers */
        send_response(fd, 204, "text/plain", "", 0);
    } else {
        send_error(fd, 404, "Not found");
    }

    free(buf);
    close(fd);
    return NULL;
}

/* ── Accept loop ─────────────────────────────────────────────── */

static void *accept_loop(void *arg) {
    SRHNServer *srv = (SRHNServer *)arg;
    signal(SIGPIPE, SIG_IGN);

    while (srv->running) {
        struct sockaddr_in cli; socklen_t cli_len = sizeof(cli);
        int conn_fd = accept(srv->listen_fd, (struct sockaddr *)&cli, &cli_len);
        if (conn_fd < 0) {
            if (errno == EINTR || errno == EBADF) break;  /* signal or stop */
            if (srv->running) perror("[api] accept");
            continue;
        }

        /* Set timeout */
        struct timeval tv = {10, 0};
        setsockopt(conn_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(conn_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        ConnArgs *ca = (ConnArgs *)malloc(sizeof(ConnArgs));
        if (!ca) { close(conn_fd); continue; }
        ca->fd  = conn_fd;
        ca->net = srv->net;

        pthread_t t;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        pthread_attr_setstacksize(&attr, 16 * 1024 * 1024);  /* 16MB stack for srhn4_query */
        if (pthread_create(&t, &attr, handle_conn, ca) != 0) {
            free(ca); close(conn_fd);
        }
        pthread_attr_destroy(&attr);
    }
    return NULL;
}

/* ── Public API ──────────────────────────────────────────────── */

SRHNServer *srhn4_server_create(SRHNNetwork4 *net, int port) {
    SRHNServer *srv = (SRHNServer *)calloc(1, sizeof(SRHNServer));
    if (!srv) return NULL;
    srv->net  = net;
    srv->port = port > 0 ? port : API_DEFAULT_PORT;

    srv->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (srv->listen_fd < 0) { free(srv); return NULL; }

    int one = 1;
    setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr = {0};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((uint16_t)srv->port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(srv->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[api] bind"); close(srv->listen_fd); free(srv); return NULL;
    }
    if (listen(srv->listen_fd, API_MAX_CONN) < 0) {
        perror("[api] listen"); close(srv->listen_fd); free(srv); return NULL;
    }

    fprintf(stderr, "[api] SRHN v3 server listening on port %d\n", srv->port);
    return srv;
}

void srhn4_server_run(SRHNServer *srv) {
    if (!srv) return;
    srv->running = true;
    accept_loop(srv); /* blocking — call in own thread if needed */
    /* Brief drain: let detached handler threads finish before caller destroys net */
    struct timespec ts = {0, 100000000L};  /* 100ms */
    nanosleep(&ts, NULL);
}

void srhn4_server_stop(SRHNServer *srv) {
    if (!srv) return;
    srv->running = false;
    shutdown(srv->listen_fd, SHUT_RDWR);
    close(srv->listen_fd);
}

void srhn4_server_free(SRHNServer *srv) {
    if (!srv) return;
    srhn4_server_stop(srv);
    free(srv);
}
