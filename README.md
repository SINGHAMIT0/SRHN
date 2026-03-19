# SRHN v4 — Production Deployment Guide

**Semantic Resonance Hypergraph Network** — Zero-dependency C11 reasoning engine.

```
102/102 tests  ·  C11  ·  ~249KB library  ·  PAC-Bayes bounds  ·  TB-scale
```

---

## Quick start (5 minutes)

```bash
# 1. Unzip and build
unzip srhn_v4_production.zip && cd srhn_v4
make all                           # → build/libsrhn4.a  srhn4_train  srhn4_test

# 2. Test (102/102 must pass)
make test

# 3. Train the network
./build/srhn4_train --epochs 5     # writes /tmp/srhn4_state.json

# 4. Start REST API
./build/srhn4_train --serve 8765 &

# 5. Open the visualizer
python3 -m http.server 3000 &
# → http://localhost:3000/srhn_v4_visualizer.html
# → click ⚡ API to connect live
```

---

## Deploy with Docker (one command)

```bash
docker-compose up -d
# API:        http://localhost:8765
# Visualizer: http://localhost:3000/visualizer.html
# Health:     curl http://localhost:8765/health
```

---

## GitHub Codespace (one-click)

Push to GitHub → **Code → Codespaces → New codespace**
The `.devcontainer/devcontainer.json` auto-runs `make all` on startup.
Ports 8765 and 3000 are forwarded automatically.

---

## Convert your documents into training data

```bash
# Any PDF, HTML, Markdown, DOCX, EPUB, or plain text
python3 srhn_docconv.py paper.pdf --domain science
python3 srhn_docconv.py *.txt *.md --domain code --output /tmp/mydata/
python3 srhn_docconv.py --url https://example.com/article
python3 srhn_docconv.py --dir ./corpus/ --recursive --domain medicine

# Output files:
#   srhn4_corpus.txt          ← stream into trainer
#   srhn4_train_dataset.json  ← Python client ingest
#   srhn_v4_train_custom.c    ← compile directly into trainer
#   srhn4_samples.csv         ← review extracted concepts

# Ingest corpus:
./build/srhn4_train --ingest /tmp/mydata/srhn4_corpus.txt
```

---

## REST API reference

All endpoints accept/return JSON. Start: `./build/srhn4_train --serve 8765`

### POST /query
```bash
curl -X POST http://localhost:8765/query \
  -H 'Content-Type: application/json' \
  -d '{"q": "what is Newton second law"}'
```
Response fields (v4):
```json
{
  "response": "...",
  "confidence": 0.87,
  "confidence_lb": 0.79,   // [R7] PAC-Bayes lower bound
  "confidence_ub": 0.96,   // [R7] PAC-Bayes upper bound
  "msg_entropy": 3.42,     // [R2] activation distribution entropy
  "max_attn_score": 0.73,  // [R3] strongest hyperedge attention
  "n_activated": 45,
  "n_chains": 8,
  "top_concepts": ["force","mass","acceleration","Newton"],
  "chains": [...],
  "latency_ms": 2.3,
  "used_llm": false
}
```

### POST /feedback
```bash
curl -X POST http://localhost:8765/feedback \
  -d '{"query_id": 1, "reward": 0.9}'
```

### POST /ingest
```bash
curl -X POST http://localhost:8765/ingest \
  -d '{"text": "Photosynthesis uses sunlight to synthesize glucose from CO2.", "source_id": 1}'
```

### GET /stats
Returns full v4 statistics including `ae_runs`, `spectral_runs`, `avg_pacbayes_bound`.

### GET /graph
Returns complete graph JSON for the visualizer.

### GET /health
Returns `{"ok": true, "version": "4.0.0"}`.

---

## Python client

```python
from srhn_client import SRHNv4Client
c = SRHNv4Client("http://localhost:8765")

# Query with full v4 metrics
r = c.query("what is quantum entanglement")
print(r['confidence'], r['confidence_lb'], r['confidence_ub'])  # PAC-Bayes
print(r['msg_entropy'], r['max_attn_score'])                    # R2, R3

# Pretty print with PAC-Bayes confidence bar
c.pretty_query("explain backpropagation")

# Benchmark
c.benchmark()

# Ingest a file
c.ingest_file("paper.txt")

# Interactive REPL
c.repl()
```

---

## Interactive trainer CLI

```bash
./build/srhn4_train --interactive

srhn4-train> run 10           # train 10 epochs
srhn4-train> eval Newton law  # single query
srhn4-train> bench            # full benchmark
srhn4-train> chains force     # show reasoning chains
srhn4-train> causal F=ma      # causal path query
srhn4-train> pagerank         # run PageRank
srhn4-train> ae               # [R4] run autoencoder pass
srhn4-train> spectral         # [R8] recompute Laplacian
srhn4-train> vae physics 3    # [R6] generate 3 VAE candidates
srhn4-train> ingest /path/corpus.txt
srhn4-train> stats            # full v4 statistics
srhn4-train> save /tmp/mynet.bin
srhn4-train> serve 8765       # start REST server
```

---

## Scalability for large datasets

```bash
# Streaming ingest — bounded RAM regardless of file size
./build/srhn4_train --ingest /path/to/100GB_corpus.txt

# Or from Python (line-at-a-time, 64KB chunks):
python3 -c "
from srhn_client import SRHNv4Client
c = SRHNv4Client()
c.ingest_file('/path/to/large.txt')
"

# Or via C API directly:
srhn_scale_ingest_path(net, "/path/corpus.txt", source_id);
```

**Scale module guarantees:**
- RAM: max 2GB nodes + 1GB edges even for TB-scale corpora
- LRU: 64K hot nodes always in RAM, cold nodes paged
- WAL: auto-rotates at 512MB, snapshots before truncate
- Memory pressure: emergency prune at 85% RAM
- BFS: bitset pool for graphs > 100K nodes (avoids per-call malloc)

---

## Visualizer keyboard shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+F` | Search and fly to node |
| Double-click | Fly to node (smooth animation) |
| `Space` | Pause/resume physics |
| `C` | Center all nodes |
| `R` | Reset layout |
| `+` / `-` | Zoom in/out |
| `0` | Reset zoom |
| `Esc` | Deselect / close search |

**Minimap:** Click anywhere on the minimap (bottom-right) to pan the viewport.

---

## Research modules (enable/disable at runtime)

```c
net->use_multisem       = true;   // R1: 3x128 orthogonal views (Gram-Schmidt)
net->use_msgpass        = true;   // R2: attention-weighted signature blending
net->use_hyperedge_attn = true;   // R3: softmax per-member scoring
net->use_autoenc        = true;   // R4: 384→192→384 AE quality filter
net->use_temporal       = true;   // R5: exp(-λΔt) edge decay, half-life 58min
net->use_vae            = false;  // R6: generative VAE (enable manually)
net->use_pacbayes       = true;   // R7: theoretical confidence bounds
net->use_spectral       = true;   // R8: Laplacian eigenvector seed reranking
```

---

## File structure

```
srhn_v4/
├── include/
│   ├── srhn_v4.h                  Master header (all types + 60+ decls)
│   ├── srhn_v4_train.h            Training pipeline API
│   └── srhn_scale.h               Scalability module API
├── src/
│   ├── srhn_core4.c               19+8 step query pipeline
│   ├── srhn_multisem.c            R1 views + R2 msg passing
│   ├── srhn_hyperedge_attn.c      R3 softmax attention
│   ├── srhn_autoenc.c             R4 AE filter + R5 temporal decay
│   ├── srhn_vae.c                 R6 generative VAE
│   ├── srhn_pacbayes_spectral.c   R7 PAC-Bayes + R8 spectral
│   ├── srhn_scale.c               Scalability: LRU, streaming, WAL, mmap
│   ├── srhn_embed4.c              GloVe/FastText/n-gram embeddings
│   ├── srhn_hnsw4.c               HNSW ANN index O(log n)
│   ├── srhn_reasoning4.c          Chains, causal A*, contradictions
│   ├── srhn_feedback4.c           Hebbian + PageRank
│   ├── srhn_memory4.c             Episodic + working memory
│   ├── srhn_persist4.c            WAL + save/load + JSON export
│   ├── srhn_utils4.c              Lang detect, selfgrow, KB
│   ├── srhn_llm4.c                llama.cpp bridge
│   ├── srhn_api4.c                Zero-dep REST server
│   ├── srhn_tokenizer4.c          BPE tokenizer
│   └── srhn_v4_train*.c           Training pipeline + REPL
├── tests/srhn4_test.c             102-test suite
├── srhn_v4_visualizer.html        Single-file graph visualizer
├── srhn_client.py                 Python REST client
├── srhn_docconv.py                Document → training data converter
├── Dockerfile                     Multi-stage production container
├── docker-compose.yml             Full stack (API + nginx)
├── nginx.conf                     Visualizer proxy config
├── .devcontainer/devcontainer.json One-click Codespace
├── scripts/verify.sh              Full system verification
├── docs/index.html                Developer documentation
├── Makefile
└── README.md
```

---

## Build targets

```bash
make all       # build everything (default)
make test      # run 102 tests
make debug     # AddressSanitizer + UBSan build
make release   # -O3 -march=native -flto
make clean     # remove build/
make docker    # build container image
```

---

## LLM Integration Guide

SRHN v4 supports three LLM backends. Set ONE of these before starting:

### Option A — OpenAI-compatible API (GPT-4o, Claude, Groq, etc.)
```bash
export OPENAI_API_KEY="sk-..."
export SRHN_LLM_BASE_URL="https://api.openai.com/v1"   # or any compatible API
export SRHN_LLM_MODEL="gpt-4o-mini"                     # default

./build/srhn4_train --epochs 5 --serve 8765
```

### Option B — Ollama (local Llama-3, Mistral, Gemma, Phi-3, etc.)
```bash
# 1. Install Ollama: https://ollama.ai
ollama pull llama3
ollama serve

# 2. Start SRHN (auto-detects Ollama)
export OLLAMA_HOST="http://localhost:11434"
export SRHN_LLM_MODEL="llama3"
./build/srhn4_train --epochs 5 --serve 8765
```

### Option C — Local llama.cpp model file
```bash
# Build llama.cpp and copy libllama.so
export SRHN_LLAMA_SO="/path/to/libllama.so"
./build/srhn4_train --llm /path/to/model.gguf --epochs 5 --serve 8765
```
**Requires:** llama.cpp >= b3000 (uses llama_decode/llama_batch API).
**Chat format** is auto-detected from filename: llama-3, mistral, gemma, phi-3, chatml, llama-2.

### Option D — No LLM (structured fallback)
```bash
./build/srhn4_train --epochs 5 --serve 8765
# Responses use graph chains + formulas. Full PAC-Bayes, R1-R8 still active.
```

### Fixed in this release vs any prior build
- Top-p + temperature + repeat-penalty sampling (was argmax-only — repetitive output)
- Actual vocab size queried per model (was hardcoded 32000 — buffer overread on Llama-3/Qwen)
- llama_decode + llama_batch API (was deprecated llama_eval — crashes on b3000+)
- Per-model mutex — no global state (was g_llama global — thread races)
- Heap prompt allocation (was stack array — thread stack overflow)
- Prompt injection sanitisation (strips BiDi overrides, chat delimiters)
- Hallucination gate: only grows graph when confidence > 50% AND grounded
