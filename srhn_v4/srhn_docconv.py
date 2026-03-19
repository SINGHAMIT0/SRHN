#!/usr/bin/env python3
"""
srhn_docconv.py  —  Document → SRHN Training Data Converter
=============================================================
Converts PDF, HTML, Markdown, plain text, EPUB, and DOCX into
SRHN v4 training samples ready for srhn4_train.

Supported input formats:
  • PDF          (via pdfminer.six or pdftotext or PyMuPDF)
  • HTML / XML   (via BeautifulSoup4 or stdlib html.parser)
  • Markdown     (via mistune or regex fallback)
  • Plain text   (.txt, .csv, .tsv, .log)
  • DOCX         (via python-docx or zipfile fallback)
  • EPUB         (via ebooklib or zipfile fallback)

Output:
  • SRHN JSON training file (srhn_v4_train_dataset.json)
  • SRHN C training include (srhn_v4_train_custom.c)  ← plug into trainer
  • Optional: CSV of extracted concept pairs

Usage:
  python3 srhn_docconv.py document.pdf
  python3 srhn_docconv.py *.pdf *.txt --domain science --output /tmp/
  python3 srhn_docconv.py --stdin --domain code < mycode.py
  python3 srhn_docconv.py --url https://example.com/paper
  python3 srhn_docconv.py --dir ./corpus/ --recursive --domain medical

  # Then use directly with trainer:
  ./build/srhn4_train --ingest /tmp/srhn_v4_corpus.txt
  # OR load the JSON:
  python3 srhn_client.py --ingest /tmp/srhn_v4_train_dataset.json
"""

import os, sys, re, json, argparse, hashlib, unicodedata
import textwrap, logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass, field, asdict
from collections import Counter

logging.basicConfig(level=logging.INFO, format="[docconv] %(message)s")
log = logging.getLogger("docconv")

# ── Concept extraction configuration ─────────────────────────────
STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","need","to","of","in","for","on","with","at","by","from",
    "as","into","through","during","before","after","above","below","between",
    "out","off","over","under","again","further","then","once","here","there",
    "when","where","why","how","all","both","each","few","more","most","some",
    "such","no","nor","not","only","same","so","than","too","very","just",
    "but","and","or","if","about","up","its","it","this","that","these",
    "those","i","we","you","he","she","they","me","him","her","us","them",
    "my","our","your","his","their","what","also","however","therefore",
    "thus","hence","although","though","while","since","because","unless",
    "until","whether","even","still","already","yet","soon","often","usually",
}

DOMAIN_KEYWORDS = {
    "science":  ["theory","law","equation","experiment","observation","hypothesis"],
    "medicine": ["patient","diagnosis","treatment","symptom","disease","drug","therapy"],
    "code":     ["function","class","method","variable","loop","algorithm","data"],
    "math":     ["theorem","proof","formula","equation","matrix","vector","integral"],
    "history":  ["century","war","empire","treaty","revolution","period","event"],
    "physics":  ["force","energy","momentum","velocity","mass","charge","field"],
    "biology":  ["cell","protein","gene","dna","enzyme","organism","evolution"],
    "law":      ["statute","regulation","clause","jurisdiction","plaintiff","defendant"],
    "finance":  ["market","equity","bond","asset","portfolio","revenue","margin"],
}

# ── Data structures ───────────────────────────────────────────────

@dataclass
class Concept:
    text: str
    score: float         # TF-IDF-like importance score
    frequency: int
    positions: List[int] = field(default_factory=list)

@dataclass
class TrainSample:
    query: str
    concepts: List[str]
    causal_rule: str
    keywords: List[str]
    domain: str
    lang: str
    difficulty: float
    source: str
    doc_id: str

@dataclass
class ConversionResult:
    samples: List[TrainSample]
    plain_text: str
    concepts: List[Concept]
    domain_detected: str
    lang_detected: str
    n_words: int
    n_sentences: int
    source_file: str

# ── Text extraction ────────────────────────────────────────────────

class TextExtractor:
    """Universal text extractor supporting multiple formats."""

    def extract(self, path: str) -> str:
        path = str(path)
        ext = Path(path).suffix.lower()
        if   ext == ".pdf":  return self._pdf(path)
        elif ext in (".html",".htm",".xml"): return self._html(path)
        elif ext in (".md",".markdown"): return self._markdown(path)
        elif ext == ".docx": return self._docx(path)
        elif ext == ".epub": return self._epub(path)
        else:                return self._plaintext(path)

    def extract_url(self, url: str) -> str:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=15) as r:
                raw = r.read().decode("utf-8", errors="replace")
            return self._html_str(raw)
        except Exception as e:
            log.error(f"URL fetch failed: {e}")
            return ""

    # ── PDF ─────────────────────────────────────────────────────
    def _pdf(self, path: str) -> str:
        # Try pdfminer.six first (best quality)
        try:
            from pdfminer.high_level import extract_text
            return extract_text(path) or ""
        except ImportError: pass

        # Try PyMuPDF (fitz)
        try:
            import fitz
            doc = fitz.open(path)
            return "\n".join(p.get_text() for p in doc)
        except ImportError: pass

        # Try pdftotext subprocess
        try:
            import subprocess
            r = subprocess.run(["pdftotext", path, "-"], capture_output=True, timeout=30)
            if r.returncode == 0: return r.stdout.decode("utf-8", errors="replace")
        except (FileNotFoundError, subprocess.TimeoutExpired): pass

        log.warning("No PDF library found. Install: pip install pdfminer.six")
        return ""

    # ── HTML ─────────────────────────────────────────────────────
    def _html(self, path: str) -> str:
        with open(path, encoding="utf-8", errors="replace") as f:
            return self._html_str(f.read())

    def _html_str(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script","style","nav","footer","header","aside"]):
                tag.decompose()
            return soup.get_text(" ", strip=True)
        except ImportError: pass
        # Stdlib fallback: strip tags with regex
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL|re.I)
        text = re.sub(r"<style[^>]*>.*?</style>",  " ", text, flags=re.DOTALL|re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        return " ".join(text.split())

    # ── Markdown ─────────────────────────────────────────────────
    def _markdown(self, path: str) -> str:
        with open(path, encoding="utf-8", errors="replace") as f:
            md = f.read()
        # Strip markdown syntax
        md = re.sub(r"```.*?```", " ", md, flags=re.DOTALL)
        md = re.sub(r"`[^`]+`", " ", md)
        md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
        md = re.sub(r"\*(.+?)\*", r"\1", md)
        md = re.sub(r"#+\s*", "", md)
        md = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", md)
        md = re.sub(r"^\s*[-*+]\s+", "", md, flags=re.M)
        return md

    # ── DOCX ─────────────────────────────────────────────────────
    def _docx(self, path: str) -> str:
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError: pass
        # Fallback: extract XML from ZIP
        import zipfile
        try:
            with zipfile.ZipFile(path) as z:
                with z.open("word/document.xml") as f:
                    xml = f.read().decode("utf-8", errors="replace")
            return re.sub(r"<[^>]+>", " ", xml)
        except Exception:
            return ""

    # ── EPUB ─────────────────────────────────────────────────────
    def _epub(self, path: str) -> str:
        import zipfile
        parts = []
        try:
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.endswith((".html",".htm",".xhtml")):
                        with z.open(name) as f:
                            parts.append(self._html_str(f.read().decode("utf-8","replace")))
        except Exception as e:
            log.warning(f"EPUB extract failed: {e}")
        return "\n".join(parts)

    def _plaintext(self, path: str) -> str:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()


# ── Language detection ─────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Simple language detection — returns SRHN language code string."""
    sample = text[:2000]
    # Script-range detection
    hi_count = sum(1 for c in sample if '\u0900' <= c <= '\u097F')
    ar_count = sum(1 for c in sample if '\u0600' <= c <= '\u06FF')
    zh_count = sum(1 for c in sample if '\u4E00' <= c <= '\u9FFF')
    if hi_count > 20: return "LANG_HINDI"
    if ar_count > 20: return "LANG_ARABIC"
    if zh_count > 20: return "LANG_CHINESE"
    # Stopword-based
    words = re.findall(r'\b[a-z]+\b', sample.lower())
    en_hits = sum(1 for w in words if w in {"the","is","are","of","and","to","in"})
    es_hits = sum(1 for w in words if w in {"el","la","los","las","es","son","de","en"})
    fr_hits = sum(1 for w in words if w in {"le","la","les","de","du","est","et"})
    de_hits = sum(1 for w in words if w in {"der","die","das","ist","und","in","von"})
    counts = {"LANG_ENGLISH":en_hits,"LANG_SPANISH":es_hits,
              "LANG_FRENCH":fr_hits,"LANG_GERMAN":de_hits}
    return max(counts, key=lambda k: counts[k])


# ── Domain detection ───────────────────────────────────────────────

def detect_domain(text: str) -> str:
    text_lower = text.lower()
    scores = {domain: sum(text_lower.count(kw) for kw in kws)
              for domain, kws in DOMAIN_KEYWORDS.items()}
    return max(scores, key=lambda k: scores[k])


# ── Concept extraction (TF-IDF-inspired) ──────────────────────────

def extract_concepts(text: str, max_concepts: int = 200) -> List[Concept]:
    """Extract significant concepts using TF-IDF-like scoring."""
    # Sentence-level tokenisation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    n_docs = max(len(sentences), 1)

    # Word frequencies
    word_re = re.compile(r'\b[a-zA-Z][a-zA-Z\-]{2,40}\b')
    doc_freq: Counter = Counter()
    term_freq: Counter = Counter()
    word_positions: Dict[str, List[int]] = {}
    pos = 0

    for sent_idx, sent in enumerate(sentences):
        words = [w.lower() for w in word_re.findall(sent)]
        seen_in_sent = set()
        for word in words:
            if word in STOPWORDS or len(word) < 3: pos += 1; continue
            term_freq[word] += 1
            if word not in seen_in_sent:
                doc_freq[word] += 1
                seen_in_sent.add(word)
            if word not in word_positions: word_positions[word] = []
            if len(word_positions[word]) < 5: word_positions[word].append(pos)
            pos += 1

    # TF-IDF-like score: tf × log(N/df)
    import math
    scored = []
    for word, tf in term_freq.most_common(500):
        if word in STOPWORDS: continue
        df = doc_freq.get(word, 1)
        score = tf * math.log(n_docs / df + 1)
        scored.append(Concept(
            text=word, score=score, frequency=tf,
            positions=word_positions.get(word, [])
        ))

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:max_concepts]


# ── Sentence-level training sample generation ─────────────────────

def sentences_to_samples(text: str, concepts: List[Concept],
                          domain: str, lang: str,
                          source: str, doc_id: str,
                          max_samples: int = 500) -> List[TrainSample]:
    """Convert text into SRHN training samples by sentence analysis."""
    concept_set = {c.text for c in concepts[:50]}  # top-50 most important
    concept_score = {c.text: c.score for c in concepts}

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
    sentences = [s.strip() for s in sentences if 10 <= len(s) <= 500]

    samples: List[TrainSample] = []
    word_re = re.compile(r'\b[a-zA-Z][a-zA-Z\-]{2,40}\b')

    for sent in sentences:
        if len(samples) >= max_samples: break
        words = [w.lower() for w in word_re.findall(sent)]

        # Find concepts mentioned in this sentence
        sent_concepts = [w for w in words
                        if w in concept_set and w not in STOPWORDS]
        # Deduplicate preserving order
        seen = set()
        unique_concepts = [c for c in sent_concepts if not (c in seen or seen.add(c))]
        if len(unique_concepts) < 2: continue

        # Extract keywords (top-4 by importance score)
        keywords = sorted(unique_concepts[:8],
                         key=lambda w: concept_score.get(w, 0), reverse=True)[:4]

        # Build causal rule from sentence structure
        causal_rule = ""
        # Look for causal patterns
        patterns = [
            (r'(\w+)\s+causes?\s+(\w+)', "{0} → causes → {1}"),
            (r'(\w+)\s+leads?\s+to\s+(\w+)', "{0} → leads to → {1}"),
            (r'(\w+)\s+results?\s+in\s+(\w+)', "{0} → results in → {1}"),
            (r'(\w+)\s+increases?\s+(\w+)', "{0} → increases → {1}"),
            (r'(\w+)\s+decreases?\s+(\w+)', "{0} → decreases → {1}"),
            (r'(\w+)\s+enables?\s+(\w+)', "{0} → enables → {1}"),
            (r'if\s+(\w+)\s+then\s+(\w+)', "if {0} → then {1}"),
            (r'(\w+)\s+requires?\s+(\w+)', "{0} → requires → {1}"),
        ]
        for pattern, template in patterns:
            m = re.search(pattern, sent.lower())
            if m:
                a, b = m.group(1), m.group(2)
                if a not in STOPWORDS and b not in STOPWORDS and len(a) > 2 and len(b) > 2:
                    causal_rule = template.format(a, b)
                    break

        # Compute difficulty: longer concepts and deeper sentences are harder
        avg_len = sum(len(c) for c in unique_concepts) / max(len(unique_concepts), 1)
        difficulty = round(min(0.3 + avg_len / 40.0 + len(unique_concepts) / 20.0, 0.9), 1)

        # Use the sentence as the query
        query = sent[:255]

        samples.append(TrainSample(
            query=query,
            concepts=unique_concepts[:8],
            causal_rule=causal_rule,
            keywords=keywords,
            domain=domain,
            lang=lang,
            difficulty=difficulty,
            source=source,
            doc_id=doc_id,
        ))

    return samples


# ── Concept-pair training sample generation ───────────────────────

def concepts_to_samples(concepts: List[Concept], domain: str, lang: str,
                         source: str, doc_id: str) -> List[TrainSample]:
    """Generate training samples from pairs of co-occurring top concepts."""
    samples = []
    top = concepts[:40]

    for i, ca in enumerate(top):
        for cb in top[i+1:i+6]:
            if abs(ca.positions[0] - cb.positions[0]) > 500: continue
            query = f"what is the relationship between {ca.text} and {cb.text}"
            concepts_list = [ca.text, cb.text]
            keywords = [ca.text, cb.text]
            samples.append(TrainSample(
                query=query,
                concepts=concepts_list,
                causal_rule=f"{ca.text} ↔ {cb.text}",
                keywords=keywords,
                domain=domain,
                lang=lang,
                difficulty=0.4,
                source=source,
                doc_id=doc_id,
            ))
    return samples[:100]


# ── C training file generator ─────────────────────────────────────

C_TEMPLATE_HEADER = """\
/*
 * srhn_v4_train_custom.c  —  Auto-generated training data
 * Generated by srhn_docconv.py from {n_sources} source(s)
 * Samples: {n_samples}  Domain: {domain}
 *
 * To use: add this file to Makefile and call srhn_train_load_custom()
 * in srhn_v4_train_main.c after srhn_train_load_builtin().
 */
#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4_train.h"
#include <string.h>

void srhn_train_load_custom(TrainDataset *ds) {{
    if (!ds) return;
"""

C_SAMPLE_TEMPLATE = """\
    /* [{idx}] {source} */
    if (ds->n_samples < TRAIN_MAX_SAMPLES) {{
        TrainSample *s = &ds->samples[ds->n_samples++];
        memset(s, 0, sizeof(*s));
        strncpy(s->query,  {query_c}, 255);
        strncpy(s->domain, {domain_c}, 31);
        s->lang = {lang};
        s->difficulty = {diff}f;
        s->expected_confidence = 0.25f;
        s->node_type = NODE_CONCEPT;
        {concept_lines}
        {keyword_lines}
        {causal_line}
    }}
"""

def escape_c_string(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace('\n', '\\n')
    return f'"{s[:255]}"'

def samples_to_c(samples: List[TrainSample], domain: str,
                  n_sources: int) -> str:
    lines = [C_TEMPLATE_HEADER.format(
        n_sources=n_sources, n_samples=len(samples), domain=domain)]

    for idx, s in enumerate(samples[:400]):  # cap at 400 samples per file
        concept_lines = ""
        for ci, c in enumerate(s.concepts[:8]):
            concept_lines += f'        if (s->n_concepts < TRAIN_MAX_CONCEPTS) strncpy(s->concepts[s->n_concepts++], {escape_c_string(c)}, MAX_WORD_LEN-1);\n'

        keyword_lines = ""
        for ki, k in enumerate(s.keywords[:4]):
            keyword_lines += f'        if (s->n_keywords < TRAIN_MAX_KEYWORDS) strncpy(s->keywords[s->n_keywords++], {escape_c_string(k)}, MAX_WORD_LEN-1);\n'

        causal_line = ""
        if s.causal_rule:
            causal_line = f'        strncpy(s->causal_rule, {escape_c_string(s.causal_rule)}, 255);'

        lines.append(C_SAMPLE_TEMPLATE.format(
            idx=idx,
            source=s.source[:40],
            query_c=escape_c_string(s.query),
            domain_c=escape_c_string(s.domain),
            lang=s.lang,
            diff=s.difficulty,
            concept_lines=concept_lines.strip(),
            keyword_lines=keyword_lines.strip(),
            causal_line=causal_line,
        ))

    lines.append(f'    printf("[custom] Loaded %u custom samples\\n", ds->n_samples);\n}}\n')
    return "".join(lines)


# ── Main conversion pipeline ─────────────────────────────────────

class DocConverter:

    def __init__(self, domain: str = "auto", lang: str = "auto",
                 max_samples: int = 500):
        self.extractor   = TextExtractor()
        self.domain      = domain
        self.lang        = lang
        self.max_samples = max_samples

    def convert_file(self, path: str) -> ConversionResult:
        log.info(f"Converting: {path}")
        text = self.extractor.extract(path)
        return self._process(text, source=str(path))

    def convert_url(self, url: str) -> ConversionResult:
        log.info(f"Fetching: {url}")
        text = self.extractor.extract_url(url)
        return self._process(text, source=url)

    def convert_text(self, text: str, source: str = "stdin") -> ConversionResult:
        return self._process(text, source=source)

    def _process(self, text: str, source: str) -> ConversionResult:
        if not text.strip():
            log.warning("No text extracted")
            return ConversionResult([], "", [], "unknown", "LANG_ENGLISH", 0, 0, source)

        # Clean text
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Detect language and domain
        lang = self.lang if self.lang != "auto" else detect_language(text)
        domain = self.domain if self.domain != "auto" else detect_domain(text)

        # Extract concepts
        concepts = extract_concepts(text, max_concepts=200)
        log.info(f"  Lang: {lang}  Domain: {domain}  Concepts: {len(concepts)}")

        # Generate training samples
        samples = sentences_to_samples(text, concepts, domain, lang,
                                        source, hashlib.md5(text[:1000].encode()).hexdigest()[:8],
                                        max_samples=self.max_samples)
        # Add concept-pair samples
        samples += concepts_to_samples(concepts, domain, lang,
                                        source, hashlib.md5(text[:1000].encode()).hexdigest()[:8])
        # Deduplicate
        seen_queries: set = set()
        unique_samples = []
        for s in samples:
            key = s.query[:80].lower()
            if key not in seen_queries:
                seen_queries.add(key)
                unique_samples.append(s)

        n_words = len(text.split())
        n_sents = len(re.split(r'(?<=[.!?])\s+', text))

        log.info(f"  Generated {len(unique_samples)} training samples from {n_words} words")
        return ConversionResult(
            samples=unique_samples,
            plain_text=text,
            concepts=concepts,
            domain_detected=domain,
            lang_detected=lang,
            n_words=n_words,
            n_sentences=n_sents,
            source_file=source,
        )

    def convert_many(self, paths: List[str]) -> List[TrainSample]:
        all_samples = []
        for p in paths:
            r = self.convert_file(p)
            all_samples.extend(r.samples)
        return all_samples


# ── Output writers ─────────────────────────────────────────────────

def write_json(samples: List[TrainSample], out_path: str, domain: str):
    """Write SRHN v4 training JSON."""
    data = {
        "version": "4.0.0",
        "domain": domain,
        "n_samples": len(samples),
        "samples": [asdict(s) for s in samples],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"JSON → {out_path}")

def write_plain_corpus(results: List[ConversionResult], out_path: str):
    """Write plain text corpus for streaming ingest."""
    with open(out_path, "w") as f:
        for r in results:
            # One line per significant sentence (for streaming ingest)
            sents = re.split(r'(?<=[.!?])\s+', r.plain_text)
            for sent in sents:
                sent = sent.strip()
                if len(sent) >= 20:
                    f.write(sent + "\n")
    log.info(f"Corpus → {out_path}")

def write_c_file(samples: List[TrainSample], out_path: str, domain: str, n_sources: int):
    """Write C training file for direct inclusion in srhn4_train."""
    code = samples_to_c(samples, domain, n_sources)
    with open(out_path, "w") as f:
        f.write(code)
    log.info(f"C file → {out_path}")

def write_csv(samples: List[TrainSample], out_path: str):
    """Write CSV of extracted samples for review."""
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query","concepts","causal_rule","domain","lang","difficulty"])
        for s in samples:
            w.writerow([s.query, "|".join(s.concepts), s.causal_rule,
                        s.domain, s.lang, s.difficulty])
    log.info(f"CSV → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Convert documents to SRHN v4 training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python3 srhn_docconv.py paper.pdf
          python3 srhn_docconv.py *.pdf *.txt --domain science
          python3 srhn_docconv.py --url https://arxiv.org/abs/2503.07959
          python3 srhn_docconv.py --dir ./corpus --recursive --domain medicine
          python3 srhn_docconv.py --stdin --domain code < myfile.py

        Then ingest with trainer:
          ./build/srhn4_train --ingest /tmp/srhn4_corpus.txt
        """)
    )
    p.add_argument("files", nargs="*", help="Input file(s)")
    p.add_argument("--url",       help="Fetch and convert a URL")
    p.add_argument("--stdin",     action="store_true", help="Read from stdin")
    p.add_argument("--dir",       help="Convert all docs in a directory")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--domain",    default="auto",
                   help=f"Domain: auto, {', '.join(DOMAIN_KEYWORDS.keys())}")
    p.add_argument("--lang",      default="auto")
    p.add_argument("--output",    "-o", default="/tmp",
                   help="Output directory (default: /tmp)")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--no-c",     action="store_true", help="Skip C file generation")
    p.add_argument("--no-csv",   action="store_true", help="Skip CSV generation")
    p.add_argument("--preview",  action="store_true", help="Print first 5 samples")
    args = p.parse_args()

    conv = DocConverter(domain=args.domain, lang=args.lang, max_samples=args.max_samples)
    results: List[ConversionResult] = []

    # Collect input sources
    all_files: List[str] = list(args.files)
    if args.dir:
        dirpath = Path(args.dir)
        exts = {".pdf",".txt",".html",".md",".docx",".epub"}
        if args.recursive:
            all_files += [str(f) for f in dirpath.rglob("*") if f.suffix.lower() in exts]
        else:
            all_files += [str(f) for f in dirpath.iterdir() if f.suffix.lower() in exts]

    # Convert files
    for f in all_files:
        if not os.path.exists(f): log.warning(f"Not found: {f}"); continue
        r = conv.convert_file(f)
        if r.samples: results.append(r)

    # URL
    if args.url:
        r = conv.convert_url(args.url)
        if r.samples: results.append(r)

    # Stdin
    if args.stdin:
        text = sys.stdin.read()
        r = conv.convert_text(text, "stdin")
        if r.samples: results.append(r)

    if not results:
        print("No documents converted. Use --help for usage.")
        return 1

    # Aggregate samples
    all_samples: List[TrainSample] = []
    for r in results: all_samples.extend(r.samples)
    domain = args.domain if args.domain != "auto" else results[0].domain_detected

    # Summary
    print(f"\n{'═'*60}")
    print(f"  SRHN v4 Document Conversion Summary")
    print(f"{'─'*60}")
    print(f"  Documents:    {len(results)}")
    print(f"  Total words:  {sum(r.n_words for r in results):,}")
    print(f"  Samples:      {len(all_samples)}")
    print(f"  Domain:       {domain}")
    print(f"  Language:     {results[0].lang_detected}")
    print(f"{'─'*60}")
    print(f"  Top concepts:")
    all_concepts: List[Concept] = []
    for r in results: all_concepts.extend(r.concepts)
    top = sorted(all_concepts, key=lambda c: c.score, reverse=True)[:15]
    for c in top:
        print(f"    {c.text:<30} score={c.score:.2f} freq={c.frequency}")
    print(f"{'═'*60}\n")

    # Preview
    if args.preview:
        print("First 5 samples:")
        for s in all_samples[:5]:
            print(f"  Q: {s.query[:80]}")
            print(f"  C: {', '.join(s.concepts[:4])}")
            print(f"  R: {s.causal_rule}")
            print()

    # Write outputs
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    json_path    = str(out / "srhn4_train_dataset.json")
    corpus_path  = str(out / "srhn4_corpus.txt")
    c_path       = str(out / "srhn_v4_train_custom.c")
    csv_path     = str(out / "srhn4_samples.csv")

    write_json(all_samples, json_path, domain)
    write_plain_corpus(results, corpus_path)
    if not args.no_c:   write_c_file(all_samples, c_path, domain, len(results))
    if not args.no_csv: write_csv(all_samples, csv_path)

    print(f"Outputs written to {out}/")
    print(f"  JSON training:   {json_path}")
    print(f"  Plain corpus:    {corpus_path}  ← use with --ingest")
    if not args.no_c:  print(f"  C training file: {c_path}  ← add to Makefile")
    if not args.no_csv:print(f"  CSV review:      {csv_path}")
    print(f"\nQuick ingest:")
    print(f"  ./build/srhn4_train --ingest {corpus_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
