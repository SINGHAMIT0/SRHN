#!/usr/bin/env python3
"""
srhn_client.py — Python client for SRHN v4 REST API
Supports all v4 research module fields:
  confidence_lb / confidence_ub  [R7 PAC-Bayes]
  msg_entropy                    [R2 Distributional msg pass]
  max_attn_score                 [R3 Hyperedge attention]
  chains with attention info     [R3 + R8]
  VAE candidate management       [R6]

Usage:
    python3 srhn_client.py                       # interactive REPL
    python3 srhn_client.py --query "Newton law"  # single query
    python3 srhn_client.py --ingest corpus.txt
    python3 srhn_client.py --stats
    python3 srhn_client.py --export graph.json
    python3 srhn_client.py --vae biology         # generate VAE candidates
    python3 srhn_client.py --benchmark           # run benchmark suite

Requires: requests (or falls back to urllib)
"""

import sys, json, argparse, time, os

try:
    import requests as _req
    def _post(url, data): return json.loads(_req.post(url, json=data, timeout=30).text)
    def _get(url): return json.loads(_req.get(url, timeout=10).text)
except ImportError:
    import urllib.request, urllib.error
    def _post(url, data):
        body = json.dumps(data).encode()
        req  = urllib.request.Request(url, body, {'Content-Type':'application/json'})
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    def _get(url): return json.loads(urllib.request.urlopen(url, timeout=10).read())


class SRHNv4Client:
    """Full-featured client for the SRHN v4 REST API."""

    def __init__(self, base_url="http://localhost:8765"):
        self.base = base_url.rstrip("/")
        self._query_count = 0

    # ── Core endpoints ─────────────────────────────────────────

    def health(self):
        try: return _get(f"{self.base}/health")
        except Exception as e: return {"status":"unreachable","error":str(e)}

    def query(self, q: str) -> dict:
        """Send a query. Returns full SRHNResult4 as dict."""
        r = _post(f"{self.base}/query", {"q": q})
        self._query_count += 1
        return r

    def feedback(self, query_id: int, reward: float) -> dict:
        return _post(f"{self.base}/feedback", {"query_id": query_id, "reward": reward})

    def ingest(self, text: str, source_id: int = 0) -> dict:
        return _post(f"{self.base}/ingest", {"text": text, "source_id": source_id})

    def stats(self) -> dict:
        return _get(f"{self.base}/stats")

    def graph(self) -> dict:
        return _get(f"{self.base}/graph")

    # ── Display helpers ─────────────────────────────────────────

    def pretty_query(self, q: str, give_feedback: bool = False):
        """Query and print a formatted v4 result."""
        t0 = time.time()
        r  = self.query(q)
        ms = (time.time() - t0) * 1000

        W = 62
        print(f"\n{'─'*W}")
        print(f"  Query:      {q}")
        print(f"{'─'*W}")
        print(f"  Response:   {r.get('response','—')}")
        print(f"{'─'*W}")

        # Confidence with PAC-Bayes bounds [R7]
        conf  = r.get('confidence',0)
        lb    = r.get('confidence_lb', max(0, conf-.2))
        ub    = r.get('confidence_ub', min(1, conf+.2))
        bar_w = 30
        lb_x  = int(lb * bar_w)
        pt_x  = int(conf * bar_w)
        ub_x  = int(ub * bar_w)
        bar   = ['░'] * bar_w
        for i in range(lb_x, ub_x): bar[i] = '▒'
        if 0 <= pt_x < bar_w: bar[pt_x] = '█'
        print(f"  Confidence: {''.join(bar)} {conf*100:.1f}% [{lb*100:.0f}%–{ub*100:.0f}%] [R7]")

        # R2 + R3 metrics
        ent   = r.get('msg_entropy', 0)
        attn  = r.get('max_attn_score', 0)
        print(f"  Msg entropy:{ent:.3f} [R2]   Max attn: {attn*100:.0f}% [R3]")
        print(f"  Nodes:      {r.get('n_activated',0)}   Chains: {r.get('n_chains',0)}   "
              f"Latency: {r.get('latency_ms',ms):.1f}ms   LLM: {'yes' if r.get('used_llm') else 'no'}")

        # Reasoning chains [R8 spectral rescored]
        for i, c in enumerate(r.get('chains', [])[:3]):
            labels = c.get('labels', [])
            rels   = c.get('relations', [])
            path   = ""
            for j, lbl in enumerate(labels):
                if j > 0: path += f" -[{rels[j-1] if j-1 < len(rels) else '→'}]→ "
                path += lbl
            print(f"  Chain {i+1}:    {path} (str={c.get('total_strength',0):.2f}"
                  f"{' ⚡causal' if c.get('is_causal') else ''})")

        # Contradictions
        for c in r.get('contradictions', []):
            print(f"  ⚡ Conflict: '{c.get('a',c.get('label_a','?'))}' "
                  f"vs '{c.get('b',c.get('label_b','?'))}' "
                  f"(conf={c.get('confidence',0)*100:.0f}%)")

        # Top concepts
        if r.get('top_concepts'):
            print(f"  Concepts:   {', '.join(r['top_concepts'][:5])}")

        print()

        if give_feedback:
            try:
                fb = input("  Feedback (1=good, 0=neutral, -1=bad, Enter=skip): ").strip()
                if fb in ('1', '0', '-1'):
                    self.feedback(self._query_count, float(fb))
                    print("  Feedback submitted.")
            except (KeyboardInterrupt, EOFError):
                pass

        return r

    def print_stats(self):
        """Print v4 statistics with research module metrics."""
        s = self.stats()
        W = 52
        print(f"\n{'═'*W}")
        print(f"  SRHN v4 Statistics")
        print(f"{'─'*W}")
        print(f"  Nodes:         {s.get('total_nodes',0):>8}  (auto: {s.get('auto_nodes',0)})")
        print(f"  Edges:         {s.get('total_edges',0):>8}")
        print(f"  Fast ring:     {s.get('fast_ring_count',0):>8}")
        print(f"  Avg PageRank:  {s.get('avg_pagerank',0):>11.6f}")
        print(f"{'─'*W}")
        print(f"  Queries:       {s.get('total_queries',0):>8}  ({s.get('avg_latency_ms',0):.1f}ms avg)")
        print(f"  HNSW hits:     {s.get('hnsw_hits',0):>8}  misses: {s.get('hnsw_misses',0)}")
        print(f"  Feedback:      {s.get('feedback_score',0):>11.4f}")
        print(f"{'─'*W}  Research modules")
        print(f"  [R4] AE runs:  {s.get('ae_runs',0):>8}  promoted: {s.get('ae_nodes_promoted',0)}")
        print(f"  [R6] VAE cand: {s.get('vae_candidates_generated',0):>8}  accepted: {s.get('vae_candidates_accepted',0)}")
        print(f"  [R7] PACBayes: {s.get('avg_pacbayes_bound',0):>11.4f}  (avg bound)")
        print(f"  [R8] Spectral: {s.get('spectral_runs',0):>8}  runs")
        print(f"  [R2] MsgEntr:  {s.get('avg_msg_entropy',0):>11.3f}")
        print(f"{'═'*W}\n")

    # ── Benchmark ───────────────────────────────────────────────

    BENCH_QUERIES = [
        ("what is Newton second law",        "physics"),
        ("explain backpropagation",          "ai"),
        ("how does hash map work",           "code"),
        ("what is quantum entanglement",     "physics"),
        ("explain transformer attention",    "ai"),
        ("what is entropy thermodynamics",   "physics"),
        ("how does GAN generator work",      "ai"),
        ("what is Rust borrow checker",      "code"),
        ("how does fear affect cortisol",    "emotion"),
        ("explain Bernoulli fluid pressure", "physics"),
    ]

    def benchmark(self):
        """Run benchmark and report v4-specific metrics."""
        print(f"\n{'═'*64}")
        print(f"  SRHN v4 Benchmark")
        print(f"{'─'*64}")
        print(f"  {'Query':<38} {'Dom':<8} {'Score':>6}  {'LB–UB':>12}  {'Chains':>6}")
        print(f"{'─'*64}")

        total_score = 0; total_lb = 0; total_ub = 0; n = 0

        for q, domain in self.BENCH_QUERIES:
            try:
                r  = self.query(q)
                sc = r.get('confidence', 0)
                lb = r.get('confidence_lb', 0)
                ub = r.get('confidence_ub', 1)
                ch = r.get('n_chains', 0)
                total_score += sc; total_lb += lb; total_ub += ub; n += 1
                color = '\033[32m' if sc > .6 else ('\033[33m' if sc > .4 else '\033[31m')
                reset = '\033[0m'
                print(f"  {q[:38]:<38} {domain:<8} {color}{sc:.4f}{reset}  "
                      f"[{lb*100:.0f}%-{ub*100:.0f}%]  {ch:>5} chains")
            except Exception as e:
                print(f"  {q[:38]:<38} {domain:<8} ERROR: {e}")

        if n > 0:
            print(f"{'─'*64}")
            avg = total_score/n
            avg_lb = total_lb/n
            avg_ub = total_ub/n
            color = '\033[32m' if avg > .6 else '\033[33m'
            print(f"  {'Average':<38} {'':8} {color}{avg:.4f}\033[0m  "
                  f"[{avg_lb*100:.0f}%-{avg_ub*100:.0f}%]  PAC-Bayes")
        print(f"{'═'*64}\n")

    # ── VAE candidate management ─────────────────────────────────

    def vae_generate(self, domain: str = "", n: int = 3):
        """Generate VAE candidates and present for review."""
        r = _post(f"{self.base}/vae-generate", {"domain": domain, "n": n})
        candidates = r.get('candidates', [])
        if not candidates:
            print("No candidates generated (domain may be too dense or VAE not fitted).")
            return
        print(f"\n{len(candidates)} VAE candidates for domain '{domain or '*'}':\n")
        for i, c in enumerate(candidates):
            print(f"  [{i}] {c.get('label_suggestion','?')} "
                  f"(novelty={c.get('novelty_score',0):.3f})")

        accepted = []
        for i, c in enumerate(candidates):
            ans = input(f"\nAccept candidate [{i}] '{c.get('label_suggestion','?')}'? "
                       f"[y/N/rename]: ").strip().lower()
            if ans == 'y':
                accepted.append(i)
            elif ans not in ('n', ''):
                # Rename and accept
                new_label = ans
                _post(f"{self.base}/vae-accept", {"idx": i, "label": new_label})
                print(f"  Accepted as '{new_label}'")
                continue

            if i in accepted:
                _post(f"{self.base}/vae-accept", {"idx": i})
                print(f"  Accepted.")

        print(f"\n{len(accepted)} candidates accepted.")

    # ── File ingestion ────────────────────────────────────────────

    def ingest_file(self, path: str, source_id: int = 0, chunk_size: int = 2000):
        """Ingest a text file in chunks."""
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        size = os.path.getsize(path)
        print(f"Ingesting {path} ({size//1024}KB)...")
        with open(path) as f:
            text = f.read()

        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        total_nodes_start = self.stats().get('total_nodes', 0)

        for i, chunk in enumerate(chunks):
            r = self.ingest(chunk, source_id)
            print(f"\r  Chunk {i+1}/{len(chunks)} → {r.get('nodes','?')} nodes", end='', flush=True)

        nodes_after = self.stats().get('total_nodes', 0)
        print(f"\nDone. +{nodes_after - total_nodes_start} new nodes from {path}")

    # ── Interactive REPL ─────────────────────────────────────────

    def repl(self):
        h = self.health()
        if h.get('status') != 'ok':
            print(f"⚠  API unreachable at {self.base}")
            print(f"   Error: {h.get('error','unknown')}")
            print(f"   Start: ./build/srhn4_train --serve 8765")
            return

        ver = h.get('version', '?')
        print(f"\n╔══ SRHN v4 Python Client ═══════════════╗")
        print(f"║  API: {self.base:<34} ║")
        print(f"║  Version: {ver:<31} ║")
        print(f"╚══════════════════════════════════════════╝")
        print("Commands: <query>  stats  bench  ingest <file>  "
              "feedback <id> <reward>  quit\n")

        while True:
            try: line = input("srhn4> ").strip()
            except (EOFError, KeyboardInterrupt): break
            if not line: continue
            if line in ("quit","q","exit"): break
            elif line == "stats":   self.print_stats()
            elif line == "bench":   self.benchmark()
            elif line.startswith("ingest "): self.ingest_file(line[7:])
            elif line.startswith("feedback "):
                parts = line.split()
                if len(parts) >= 3:
                    self.feedback(int(parts[1]), float(parts[2]))
                    print("Feedback submitted.")
            else:
                self.pretty_query(line, give_feedback=True)


# ── CLI ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="SRHN v4 Python client")
    p.add_argument("--host",      default="localhost")
    p.add_argument("--port",      default=8765, type=int)
    p.add_argument("--query",     "-q", help="Single query")
    p.add_argument("--ingest",    "-i", help="Ingest a text file")
    p.add_argument("--stats",     "-s", action="store_true")
    p.add_argument("--export",    "-e", help="Export graph JSON to file")
    p.add_argument("--benchmark", "-b", action="store_true")
    p.add_argument("--vae",              help="Generate VAE candidates for domain")
    p.add_argument("--feedback",  nargs=2, metavar=("ID","REWARD"))
    args = p.parse_args()

    c = SRHNv4Client(f"http://{args.host}:{args.port}")

    if args.stats:      c.print_stats()
    elif args.query:    c.pretty_query(args.query)
    elif args.benchmark:c.benchmark()
    elif args.ingest:   c.ingest_file(args.ingest)
    elif args.vae:      c.vae_generate(args.vae)
    elif args.feedback:
        c.feedback(int(args.feedback[0]), float(args.feedback[1]))
        print("Feedback submitted.")
    elif args.export:
        g = c.graph()
        with open(args.export, "w") as f: json.dump(g, f, indent=2)
        print(f"Graph → {args.export}")
    else:
        c.repl()


if __name__ == "__main__":
    main()
