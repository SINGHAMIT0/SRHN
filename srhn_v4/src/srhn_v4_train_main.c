/*
 * srhn_v3_train_main.c  —  SRHN v3 Training CLI
 * ===============================================
 * Full port of v2 trainer to SRHN v3 APIs.
 * Exports to /tmp/srhn4_state.json (visualizer-compatible).
 *
 * Usage:
 *   ./srhn4_train [options]
 *   ./srhn4_train --epochs 20 --lr 0.04 --verbose
 *   ./srhn4_train --ingest myfile.txt --lang en
 *   ./srhn4_train --benchmark
 *   ./srhn4_train --interactive
 *   ./srhn4_train --llm /path/to/model.gguf
 *   ./srhn4_train --serve 8765        (start REST API after training)
 */
#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include "../include/srhn_v4_train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <strings.h>
#include <ctype.h>

/* Paths */
#define SAVE_PATH        "/tmp/srhn4_net.bin"
#define EXPORT_PATH      "/tmp/srhn4_state.json"    /* visualizer reads this */
#define TRAIN_JSON_PATH  "/tmp/srhn4_train.json"
#define TRAIN_CSV_PATH   "/tmp/srhn4_train.csv"
#define WAL_PATH         "/tmp/srhn4_net.bin.wal"

/* ANSI */
#define RST    "\033[0m"
#define BOLD   "\033[1m"
#define CYAN   "\033[36m"
#define GREEN  "\033[32m"
#define YELLOW "\033[33m"
#define RED    "\033[31m"
#define DIM    "\033[2m"

static volatile int g_running = 1;
static void on_sig(int s) { (void)s; g_running = 0; }

/* ── Banner ─────────────────────────────────────────────────── */
static void print_banner(void) {
    printf(CYAN BOLD
        "╔══════════════════════════════════════════════════════╗\n"
        "║   SRHN v4 — Training Pipeline                        ║\n"
        "║   R1-R8 · HNSW · Hebbian · PAC-Bayes · Spectral        ║\n"
        "╚══════════════════════════════════════════════════════╝\n"
        RST "\n");
}

/* ── Help ───────────────────────────────────────────────────── */
static void print_help_interactive(void) {
    printf(YELLOW "\nInteractive Commands:\n" RST);
    static const char *cmds[][2] = {
        {"run [N]",              "Run N training epochs (default 10)"},
        {"epoch",                "Single training epoch"},
        {"bench",                "Run full benchmark suite"},
        {"eval <query>",         "Evaluate single query"},
        {"ingest <file>",        "Ingest text file into graph"},
        {"online",               "Online learning from feedback history"},
        {"pagerank",             "Run PageRank iteration now"},
        {"stats",                "Full network + training stats"},
        {"summary",              "Training summary by domain"},
        {"save",                 "Save network binary"},
        {"export",               "Export graph JSON for visualizer"},
        {"csv",                  "Export training history CSV"},
        {"add <query>|c1|c2...", "Add custom training sample"},
        {"lr <value>",           "Set learning rate"},
        {"verbose on/off",       "Toggle verbose output"},
        {"chains <query>",       "Show reasoning chains for query"},
        {"causal <from> <to>",   "Show causal path between two concepts"},
        {"serve [port]",         "Start REST API server (blocks)"},
        {"quit",                 "Save and exit"},
        {NULL, NULL}
    };
    for (int i = 0; cmds[i][0]; i++)
        printf("  " GREEN "%-30s" RST " %s\n", cmds[i][0], cmds[i][1]);
    printf("\n");
}

/* ── Load KB ────────────────────────────────────────────────── */
static void load_all_kb(SRHNNetwork4 *net) {
    srhn4_load_physics(net);
    srhn4_load_code(net);
    printf(GREEN "Knowledge base loaded: %u nodes, %u edges\n" RST,
           net->n_nodes, net->n_edges);
}

/* ── Export helper (atomic write to visualizer path) ─────────── */
static void do_export(SRHNNetwork4 *net) {
    srhn4_export_json(net, EXPORT_PATH);
    printf(GREEN "  → Graph JSON: %s\n" RST, EXPORT_PATH);
}

/* ── Main ───────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    signal(SIGINT,  on_sig);
    signal(SIGTERM, on_sig);

    print_banner();

    /* ── Parse CLI arguments ── */
    uint32_t  epochs      = TRAIN_DEFAULT_EPOCHS;
    float     lr          = TRAIN_DEFAULT_LR;
    bool      do_bench    = false;
    bool      interactive = false;
    bool      verbose     = true;
    bool      do_serve    = false;
    int       serve_port  = API_DEFAULT_PORT;
    char      ingest_file [512] = {0};
    char      llm_path    [512] = {0};
    LangID    ingest_lang = LANG_ENGLISH;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i],"--epochs")      && i+1<argc) epochs = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i],"--lr")           && i+1<argc) lr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--benchmark"))                do_bench = true;
        else if (!strcmp(argv[i],"--interactive"))              interactive = true;
        else if (!strcmp(argv[i],"--quiet"))                    verbose = false;
        else if (!strcmp(argv[i],"--serve"))       do_serve = true;
        else if (!strcmp(argv[i],"--port")         && i+1<argc) serve_port = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--ingest")       && i+1<argc) strncpy(ingest_file, argv[++i], 511);
        else if (!strcmp(argv[i],"--llm")          && i+1<argc) strncpy(llm_path, argv[++i], 511);
        else if (!strcmp(argv[i],"--lang")         && i+1<argc) {
            const char *l = argv[++i];
            if      (!strcmp(l,"hi")) ingest_lang = LANG_HINDI;
            else if (!strcmp(l,"es")) ingest_lang = LANG_SPANISH;
            else if (!strcmp(l,"fr")) ingest_lang = LANG_FRENCH;
            else if (!strcmp(l,"de")) ingest_lang = LANG_GERMAN;
        }
        else if (!strcmp(argv[i],"--help") || !strcmp(argv[i],"-h")) {
            printf("Usage: %s [--epochs N] [--lr F] [--benchmark] [--interactive]\n"
                   "       [--ingest FILE] [--lang CODE] [--llm PATH]\n"
                   "       [--serve] [--port N] [--quiet]\n", argv[0]);
            return 0;
        }
    }

    /* ── Create / load network ── */
    SRHNNetwork4 *net = srhn4_load(SAVE_PATH);
    if (!net) {
        printf(CYAN "No saved network. Building fresh...\n" RST);
        net = srhn4_create();
        if (!net) { fprintf(stderr, "[ERROR] OOM\n"); return 1; }
        load_all_kb(net);
    } else {
        printf(GREEN "Resumed: %u nodes, %u edges\n" RST, net->n_nodes, net->n_edges);
    }

    /* Open WAL for crash safety */
    srhn4_wal_open(net, WAL_PATH);

    /* LLM init (graceful — no error if no model) */
    if (strlen(llm_path) > 0) {
        bool llm_ok = srhn4_llm_init(&net->llm, llm_path, LLM_CTX_SIZE, 4);
        printf(llm_ok ? GREEN "LLM loaded: %s\n" RST : YELLOW "LLM unavailable (fallback mode)\n" RST,
               llm_path);
    }

    /* ── Setup training state ── */
    TrainState ts;
    srhn_train_init(&ts, lr);
    ts.verbose = verbose;
    srhn_train_load_builtin(&ts.dataset);

    /* ── Ingest file if given ── */
    if (strlen(ingest_file) > 0) {
        printf(CYAN "Ingesting: %s [%s]\n" RST, ingest_file,
               srhn4_lang_name(ingest_lang));
        srhn_train_ingest_file(net, ingest_file, ingest_lang);
        srhn4_save(net, SAVE_PATH);
        do_export(net);
        srhn4_wal_close(net);
        srhn4_destroy(net);
        return 0;   /* early exit after ingest */
    }

    /* ── Benchmark-only mode ── */
    if (do_bench) {
        srhn_train_benchmark(net, &ts);
        srhn4_save(net, SAVE_PATH);
        do_export(net);
        srhn4_wal_close(net);
        srhn4_destroy(net);
        return 0;
    }

    /* ── Non-interactive batch mode ── */
    if (!interactive) {
        /* serve-only: if --serve set but no explicit --epochs, skip training+benchmark */
        if (do_serve && epochs == TRAIN_DEFAULT_EPOCHS) {
            /* Load-and-serve mode: skip training, benchmarking, CSV output */
            fprintf(stderr, "[serve] Loaded %u nodes, %u edges. Starting server...\n",
                    net->n_nodes, net->n_edges);
            srhn4_save(net, SAVE_PATH);
            do_export(net);
        } else {
            srhn_train_run(net, &ts, epochs);
            srhn_train_benchmark(net, &ts);
            srhn_train_export_csv(&ts, TRAIN_CSV_PATH);
            srhn_train_export_json(net, &ts, TRAIN_JSON_PATH);
            srhn4_print_stats(net);
            srhn4_save(net, SAVE_PATH);
            do_export(net);
        }

        if (do_serve) {
            printf(CYAN "Starting REST API on port %d...\n" RST, serve_port);
            SRHNServer4 *srv = srhn4_server_create(net, serve_port);
            if (srv) srhn4_server_run(srv); /* blocks */
        }

        srhn4_wal_close(net);
        srhn4_destroy(net);
        return 0;
    }

    /* ══ Interactive REPL ══════════════════════════════════════ */
    print_help_interactive();
    char line[1024];

    while (g_running) {
        printf(BOLD CYAN "srhn3-train> " RST);
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) break;
        size_t len = strlen(line);
        while (len > 0 && (line[len-1]=='\n'||line[len-1]=='\r')) line[--len] = '\0';
        if (len == 0) continue;

        /* ── Tokenize command ── */
        char *cmd = line;
        while (*cmd == ' ') cmd++;

        /* ── Commands ── */
        if (!strcmp(cmd,"quit")||!strcmp(cmd,"q")||!strcmp(cmd,"exit")) {
            break;

        } else if (!strcmp(cmd,"help")||!strcmp(cmd,"?")) {
            print_help_interactive();

        } else if (!strncmp(cmd,"run",3)) {
            uint32_t ep = epochs;
            if (len > 4 && isdigit((unsigned char)cmd[4])) ep = (uint32_t)atoi(cmd+4);
            printf(CYAN "Running %u epoch(s) (LR=%.4f)...\n" RST, ep, ts.learning_rate);
            srhn_train_run(net, &ts, ep);
            do_export(net);

        } else if (!strcmp(cmd,"epoch")) {
            srhn_train_epoch(net, &ts);
            do_export(net);

        } else if (!strcmp(cmd,"bench")) {
            srhn_train_benchmark(net, &ts);

        } else if (!strcmp(cmd,"online")) {
            srhn_train_online(net, &ts);

        } else if (!strcmp(cmd,"pagerank")) {
            srhn4_run_pagerank(net);
            printf(GREEN "PageRank done. avg=%.6f\n" RST, net->stats.avg_pagerank);

        } else if (!strcmp(cmd,"stats")) {
            srhn4_print_stats(net);
            printf("Training: epoch=%u  correct=%u/%u  LR=%.5f  LLM=%u calls\n",
                   ts.epoch, ts.correct, ts.total_tested,
                   ts.learning_rate, ts.llm_calls);

        } else if (!strcmp(cmd,"summary")) {
            srhn_train_print_summary(&ts);

        } else if (!strcmp(cmd,"save")) {
            srhn4_save(net, SAVE_PATH);
            printf(GREEN "Saved → %s\n" RST, SAVE_PATH);

        } else if (!strcmp(cmd,"export")) {
            do_export(net);
            srhn_train_export_json(net, &ts, TRAIN_JSON_PATH);
            printf(GREEN "Training JSON → %s\n" RST, TRAIN_JSON_PATH);

        } else if (!strcmp(cmd,"csv")) {
            srhn_train_export_csv(&ts, TRAIN_CSV_PATH);
            printf(GREEN "CSV → %s\n" RST, TRAIN_CSV_PATH);

        } else if (!strncmp(cmd,"eval ",5)) {
            const char *q = cmd+5;
            float score = srhn_train_eval(net, q, NULL, 0);
            SRHNResult4 r = srhn4_query(net, q);
            printf("Query:   \"%s\"\n", q);
            printf("Score:   %.4f  Confidence: %.4f  Resonance: %.4f\n",
                   score, r.confidence, r.best_resonance);
            printf("Chains:  %u  Contradictions: %u  LLM: %s\n",
                   r.n_chains, r.n_contradictions,
                   r.used_llm ? "yes" : "no (fallback)");
            printf("Lang:    %s  Latency: %.1fms\n",
                   srhn4_lang_name(r.response_lang), r.latency_ms);
            printf("Top:     ");
            for (int i = 0; i < r.n_top_concepts; i++)
                printf("[%s] ", r.top_concepts[i]);
            printf("\nResponse: %s\n", r.response);

        } else if (!strncmp(cmd,"chains ",7)) {
            SRHNResult4 r = srhn4_query(net, cmd+7);
            printf("Chains (%u):\n", r.n_chains);
            for (uint8_t c = 0; c < r.n_chains; c++) {
                ReasoningChain *ch = &r.chains[c];
                printf("  [%u] strength=%.3f causal=%s  ",
                       c+1, ch->total_strength, ch->is_causal ? "yes" : "no");
                for (uint8_t j = 0; j < ch->length; j++) {
                    if (j > 0) printf(" -[%s]→ ", ch->relations[j-1]);
                    printf("%s", ch->labels[j]);
                }
                printf("\n");
            }
            if (r.n_contradictions > 0) {
                printf("Contradictions (%u):\n", r.n_contradictions);
                for (uint8_t c = 0; c < r.n_contradictions; c++)
                    printf("  \"%s\" ⚡ \"%s\" (conf=%.2f)\n",
                           r.contradictions[c].label_a,
                           r.contradictions[c].label_b,
                           r.contradictions[c].confidence);
            }

        } else if (!strncmp(cmd,"causal ",7)) {
            /* causal <from_label> <to_label> */
            char *sp = strchr(cmd+7, ' ');
            if (!sp) { printf(RED "Usage: causal <from> <to>\n" RST); continue; }
            *sp = '\0';
            const char *from_str = cmd + 7;
            const char *to_str   = sp + 1;

            /* Find node IDs */
            Signature fa, tb;
            srhn4_embed_text(net, from_str, LANG_ENGLISH, &fa);
            srhn4_embed_text(net, to_str,   LANG_ENGLISH, &tb);
            uint32_t fid = UINT32_MAX, tid = UINT32_MAX;
            float bf = 0.f, bt = 0.f;
            for (uint32_t i = 0; i < net->n_nodes; i++) {
                if (net->nodes[i].type == NODE_PRUNED) continue;
                float rf = srhn4_resonance(&fa, &net->nodes[i].sig);
                float rt = srhn4_resonance(&tb, &net->nodes[i].sig);
                if (rf > bf) { bf = rf; fid = i; }
                if (rt > bt) { bt = rt; tid = i; }
            }
            if (fid == UINT32_MAX || tid == UINT32_MAX) {
                printf(RED "Concepts not found in graph.\n" RST); continue;
            }
            ReasoningChain chain;
            srhn4_causal_path(net, fid, tid, &chain);
            printf("Causal path: %s → %s\n", from_str, to_str);
            for (uint8_t j = 0; j < chain.length; j++) {
                if (j > 0) printf(" -[%s]→ ", chain.relations[j-1]);
                printf("%s(%.2f)", chain.labels[j], chain.strengths[j]);
            }
            printf("\nTotal strength: %.3f\n", chain.total_strength);

        } else if (!strncmp(cmd,"ingest ",7)) {
            srhn_train_ingest_file(net, cmd+7, LANG_ENGLISH);
            do_export(net);

        } else if (!strncmp(cmd,"lr ",3)) {
            ts.learning_rate = srhn4_clampf((float)atof(cmd+3), 0.001f, 1.f);
            printf(GREEN "LR → %.5f\n" RST, ts.learning_rate);

        } else if (!strncmp(cmd,"verbose ",8)) {
            ts.verbose = (strstr(cmd+8,"on") != NULL);
            printf("Verbose: %s\n", ts.verbose ? "ON" : "OFF");

        } else if (!strncmp(cmd,"serve",5)) {
            int port = serve_port;
            if (len > 6) port = atoi(cmd+6);
            printf(CYAN "Starting REST API on port %d...\n" RST, port);
            printf(DIM "Endpoints: POST /query  POST /feedback  GET /stats  GET /graph\n" RST);
            SRHNServer4 *srv = srhn4_server_create(net, port);
            if (srv) srhn4_server_run(srv);  /* blocks */

        } else if (!strncmp(cmd,"add ",4)) {
            /* Syntax: add <query> | c1 | c2 | ... */
            char query[256] = {0};
            char cbuf [512] = {0};
            char *pipe = strchr(cmd+4, '|');
            if (pipe) {
                size_t qlen = (size_t)(pipe - cmd - 4);
                if (qlen > 255) qlen = 255;
                strncpy(query, cmd+4, qlen);
                /* Trim trailing space */
                int qi = (int)strlen(query) - 1;
                while (qi >= 0 && query[qi] == ' ') query[qi--] = '\0';
                strncpy(cbuf, pipe+1, 511);
            } else {
                strncpy(query, cmd+4, 255);
            }
            const char *concepts[TRAIN_MAX_CONCEPTS]; int nc = 0;
            char copy[512]; strncpy(copy, cbuf, 512); copy[511] = 'char copy[512]; strncpy(copy, cbuf, 511);';
            char *tok = strtok(copy, "|,");
            while (tok && nc < TRAIN_MAX_CONCEPTS) {
                while (*tok == ' ') tok++;
                char *end = tok + strlen(tok) - 1;
                while (end > tok && *end == ' ') *end-- = '\0';
                if (*tok) concepts[nc++] = tok;
                tok = strtok(NULL, "|,");
            }
            srhn_train_add_sample(&ts.dataset, query,
                concepts, nc, NULL, NULL, 0,
                "custom", LANG_ENGLISH, 0.5f);
            printf(GREEN "[+] Sample added: \"%s\" (%d concepts)\n" RST, query, nc);

        } else {
            printf(RED "Unknown: %s\n" RST, cmd);
            printf("Type 'help' for commands.\n");
        }
    }

    /* ── Final save & export ── */
    printf(CYAN "\nFinalizing...\n" RST);
    srhn_train_print_summary(&ts);
    srhn_train_export_csv(&ts, TRAIN_CSV_PATH);
    srhn_train_export_json(net, &ts, TRAIN_JSON_PATH);
    srhn4_save(net, SAVE_PATH);
    do_export(net);

    srhn4_wal_close(net);
    srhn4_destroy(net);
    printf(GREEN "Done. Graph saved to %s\n" RST, SAVE_PATH);
    printf(GREEN "Visualizer data: %s\n" RST, EXPORT_PATH);
    return 0;
}
