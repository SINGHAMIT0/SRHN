/*
 * srhn_utils.c  —  Utilities, language detection, self-growing, domain loaders
 */

#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4.h"
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <ctype.h>
#include <strings.h>
#include <time.h>
#include <stdio.h>

/* ── Timestamp & RNG ─────────────────────────────────────────── */

uint64_t srhn4_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

uint64_t srhn4_rand(SRHNNetwork4 *net) {
    uint64_t x = net->rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return (net->rng_state = x);
}

float srhn4_randf(SRHNNetwork4 *net) {
    return (float)(srhn4_rand(net) & 0xFFFFFF) / (float)0xFFFFFF;
}

void srhn4_str_lower(char *s) {
    for (; *s; s++) *s = (char)tolower((unsigned char)*s);
}

/* ── Language detection ──────────────────────────────────────── */
/*
 * Heuristic multi-signal language detection:
 * 1. UTF-8 script byte range detection (CJK, Devanagari, Arabic, Cyrillic)
 * 2. Latin stopword matching for European languages
 * 3. Code keyword detection
 * Falls back to English for ambiguous Latin text.
 */
LangID srhn4_detect_lang(const char *text) {
    if (!text || !*text) return LANG_UNKNOWN;

    const unsigned char *u = (const unsigned char *)text;
    int devanagari = 0, arabic = 0, cjk = 0, cyrillic = 0, latin = 0;
    int total = 0;

    while (*u && total < 512) {
        uint8_t b = *u;
        if (b < 0x80) {
            if (isalpha(b)) latin++;
            total++; u++;
        } else if ((b & 0xE0) == 0xC0 && u[1]) {
            uint32_t cp = ((b & 0x1F) << 6) | (u[1] & 0x3F);
            if (cp >= 0x0900 && cp <= 0x097F) devanagari++;
            else if (cp >= 0x0600 && cp <= 0x06FF) arabic++;
            else if (cp >= 0x0400 && cp <= 0x04FF) cyrillic++;
            total++; u += 2;
        } else if ((b & 0xF0) == 0xE0 && u[1] && u[2]) {
            uint32_t cp = ((b & 0x0F) << 12) | ((u[1] & 0x3F) << 6) | (u[2] & 0x3F);
            if ((cp >= 0x4E00 && cp <= 0x9FFF) ||  /* CJK Unified */
                (cp >= 0x3040 && cp <= 0x30FF) ||  /* Hiragana/Katakana */
                (cp >= 0xAC00 && cp <= 0xD7AF))    /* Hangul */
                cjk++;
            total++; u += 3;
        } else { u++; }
    }

    if (total == 0) return LANG_UNKNOWN;

    /* Script thresholds: 15% of chars is enough to identify */
    int thresh = total / 7;
    if (devanagari > thresh) return LANG_HINDI;
    if (arabic     > thresh) return LANG_ARABIC;
    if (cjk        > thresh) return LANG_CHINESE;
    if (cyrillic   > thresh) return LANG_RUSSIAN;

    /* Code detection — check before language (code may contain any language) */
    char lower[512];
    strncpy(lower, text, 511); lower[511] = '\0';
    for (char *p = lower; *p; p++) *p = (char)tolower((unsigned char)*p);

    /* Strong code indicators */
    int code_score = 0;
    if (strstr(lower, "#include"))  code_score += 3;
    if (strstr(lower, "def "))      code_score += 2;
    if (strstr(lower, "void "))     code_score += 2;
    if (strstr(lower, "func "))     code_score += 2;
    if (strstr(lower, "class "))    code_score += 1;
    if (strstr(lower, "import "))   code_score += 1;
    if (strstr(lower, "return "))   code_score += 1;
    if (strstr(lower, "printf"))    code_score += 2;
    if (strstr(lower, "console."))  code_score += 2;
    if (strstr(lower, "->"))        code_score += 1;
    if (strstr(lower, "=>"))        code_score += 1;
    if (strstr(lower, "nullptr"))   code_score += 2;
    if (code_score >= 3) return LANG_CODE;

    /* Latin-script European languages via stopwords */
    /* Spanish — handle ¿ prefix and accented chars via substring matching */
    int es = 0;
    if (strstr(lower, " que ") || strstr(lower, "qué")) es++;
    if (strstr(lower, " los ") || strstr(lower, " las ")) es++;
    if (strstr(lower, " una ") || strstr(lower, " unos")) es++;
    if (strstr(lower, " por ") || strstr(lower, " para ")) es++;
    if (strstr(lower, "ción")  || strstr(lower, "cion")) es++;
    if (strstr(lower, " esto ") || strstr(lower, " este ")) es++;
    if (strstr(lower, "nerg")  || strstr(lower, "cin")) es++;  /* energía, cinética */
    if (strstr(lower, " es "))  es++;
    if (es >= 2) return LANG_SPANISH;

    /* French */
    int fr = 0;
    if (strstr(lower, " les ")  || strstr(lower, " la "))  fr++;
    if (strstr(lower, " des ")  || strstr(lower, " du "))  fr++;
    if (strstr(lower, " est ")  || strstr(lower, " sont ")) fr++;
    if (strstr(lower, " une ")  || strstr(lower, " un "))  fr++;
    if (strstr(lower, " pas ")  || strstr(lower, " ne "))  fr++;
    if (strstr(lower, " vous ") || strstr(lower, "aller")) fr++;
    if (strstr(lower, "bonjour")|| strstr(lower, "aujourd")) fr += 2;
    if (strstr(lower, "comment")) fr++;
    if (fr >= 2) return LANG_FRENCH;

    /* German */
    int de = 0;
    if (strstr(lower, " der ")) de++;
    if (strstr(lower, " die ")) de++;
    if (strstr(lower, " das ")) de++;
    if (strstr(lower, " und ")) de++;
    if (strstr(lower, " ist ")) de++;
    if (strstr(lower, " ich ")) de++;
    if (strstr(lower, "schaft")) de++;
    if (de >= 2) return LANG_GERMAN;

    /* Portuguese */
    int pt = 0;
    if (strstr(lower, " são ")) pt++;
    if (strstr(lower, " não ")) pt++;
    if (strstr(lower, " mas ")) pt++;
    if (strstr(lower, "ção"))   pt += 2;
    if (pt >= 2) return LANG_PORTUGUESE;

    /* Italian */
    int it = 0;
    if (strstr(lower, " gli ")) it++;
    if (strstr(lower, " che ")) it++;
    if (strstr(lower, " una ")) it++;
    if (strstr(lower, " nel ")) it++;
    if (it >= 2) return LANG_ITALIAN;

    /* Mixed: if multiple non-English signals present */
    if (es + fr + de > 2) return LANG_MIXED;

    return LANG_ENGLISH;
}

const char *srhn4_lang_name(LangID lang) {
    static const char *names[LANG_COUNT] = {
        "Unknown","English","Hindi","Spanish","French","German",
        "Arabic","Chinese","Japanese","Russian","Portuguese",
        "Italian","Bengali","Code","Mixed"
    };
    return ((unsigned)lang < LANG_COUNT) ? names[(int)lang] : "Unknown";
}

/* ── Self-growing ─────────────────────────────────────────────── */

uint32_t srhn4_selfgrow(SRHNNetwork4 *net, const char *text, LangID lang) {
    if (!text || !*text) return UINT32_MAX;

    Signature sig;
    srhn4_embed_text(net, text, lang, &sig);

    /* Check HNSW for near-duplicate */
    uint32_t nn_id; float nn_dist;
    int found = srhn4_hnsw_search(&net->hnsw, &sig, 8, &nn_id, &nn_dist, 1, net);
    if (found > 0 && (1.f - nn_dist) > 0.78f) return nn_id;

    /* Create new node */
    uint32_t nid = srhn4_add_node(net, text, NODE_CONCEPT, lang);
    if (nid == UINT32_MAX) return UINT32_MAX;

    net->nodes[nid].auto_created = true;
    net->selfgrow_count++;
    net->stats.auto_nodes++;

    /* Connect to nearest neighbor */
    if (found > 0 && nn_id < net->n_nodes && (1.f - nn_dist) > 0.15f)
        srhn4_connect(net, nid, nn_id, 1.f - nn_dist, EDGE_ASSOC);

    /* Expand synonyms for English */
    if (lang == LANG_ENGLISH) {
        char tokens[32][MAX_WORD_LEN];
        int nt = srhn4_tok_simple(text, tokens, 32);
        for (int t = 0; t < nt; t++) {
            LexEntry *le = NULL;
            for (uint32_t li = 0; li < net->lexicon.n_entries; li++) {
                if (strcasecmp(net->lexicon.entries[li].word, tokens[t]) == 0) {
                    le = &net->lexicon.entries[li]; break;
                }
            }
            if (!le) continue;
            for (uint8_t s = 0; s < le->n_synonyms; s++) {
                uint32_t sid = srhn4_selfgrow(net, le->synonyms[s], lang);
                if (sid != UINT32_MAX && sid != nid)
                    srhn4_connect(net, nid, sid, 0.78f, EDGE_SYNONYM);
            }
        }
    }
    return nid;
}

void srhn4_selfgrow_from_query(SRHNNetwork4 *net, const char *query, float best_res) {
    if (!net->selfgrow_enabled || best_res > net->selfgrow_thresh) return;

    LangID lang = srhn4_detect_lang(query);
    char tokens[32][MAX_WORD_LEN];
    int nt = srhn4_tok_simple(query, tokens, 32);

    static const char *stop[] = {
        "the","a","an","is","are","was","were","be","been","being","have","has",
        "had","do","does","did","will","would","could","should","may","might",
        "this","that","these","those","it","its","i","you","he","she","we","they",
        "in","on","at","by","for","with","about","from","to","of","and","or","but",
        "not","no","yes","if","then","so","as","than","like","just","also",NULL
    };

    uint32_t last_id = UINT32_MAX;
    for (int t = 0; t < nt; t++) {
        if (strlen(tokens[t]) < 3) continue;
        bool is_stop = false;
        for (int s = 0; stop[s]; s++)
            if (strcmp(tokens[t], stop[s]) == 0) { is_stop = true; break; }
        if (is_stop) continue;

        uint32_t nid = srhn4_selfgrow(net, tokens[t], lang);
        if (nid != UINT32_MAX && last_id != UINT32_MAX && last_id != nid)
            srhn4_connect(net, last_id, nid, 0.55f, EDGE_ASSOC);
        last_id = nid;
    }
}

/* ── Document ingestion ──────────────────────────────────────── */
/*
 * srhn4_ingest_document:
 *   Tokenize text → embed each word/phrase → find or create graph node
 *   → connect semantically similar nodes above threshold
 *   Enables the graph to grow from raw text at scale.
 */
void srhn4_ingest_document(SRHNNetwork4 *net, const char *text, uint32_t source_id) {
    if (!text || !*text) return;

    LangID lang = srhn4_detect_lang(text);
    char tokens[128][MAX_WORD_LEN];
    int nt = srhn4_tok_simple(text, tokens, 128);

    static const char *stop[] = {
        "the","a","an","is","are","was","were","be","been","being","have","has",
        "had","do","does","did","will","would","could","should","this","that",
        "in","on","at","by","for","with","about","from","to","of","and","or",
        "but","not","it","its","i","you","he","she","we","they","so","as",NULL
    };

    /* Build sliding window of significant tokens and create/connect nodes */
    uint32_t window[8]; int wn = 0;

    for (int t = 0; t < nt; t++) {
        if (strlen(tokens[t]) < 4) continue;
        bool is_stop = false;
        for (int s = 0; stop[s]; s++)
            if (strcmp(tokens[t], stop[s]) == 0) { is_stop = true; break; }
        if (is_stop) continue;

        /* Check existing nodes via HNSW */
        Signature tsig;
        srhn4_embed_text(net, tokens[t], lang, &tsig);

        uint32_t nn_id; float nn_dist;
        int found = srhn4_hnsw_search(&net->hnsw, &tsig, 4, &nn_id, &nn_dist, 1, net);

        uint32_t nid;
        if (found > 0 && (1.f - nn_dist) > 0.88f) {
            /* Use existing node */
            nid = nn_id;
            net->nodes[nid].usage_count += 0.5f;
        } else {
            /* Create new node */
            nid = srhn4_add_node(net, tokens[t], NODE_CONCEPT, lang);
            if (nid == UINT32_MAX) continue;
            net->nodes[nid].auto_created = true;
            net->nodes[nid].source_id    = source_id;
            net->nodes[nid].entropy_score = 0.3f; /* low until reinforced */
            net->stats.auto_nodes++;
        }

        /* Connect to nearby window nodes */
        for (int wi = 0; wi < wn; wi++) {
            if (window[wi] == nid || window[wi] >= net->n_nodes) continue;
            float sim = srhn4_resonance(&tsig, &net->nodes[window[wi]].sig);
            if (sim > 0.25f)
                srhn4_connect(net, nid, window[wi], sim * 0.7f, EDGE_ASSOC);
        }

        /* Slide window */
        if (wn < 8) window[wn++] = nid;
        else {
            memmove(window, window + 1, 7 * sizeof(uint32_t));
            window[7] = nid;
        }
    }
}

/* ── Physics laws ────────────────────────────────────────────── */

static void add_law(SRHNNetwork4 *net, const char *name, const char *formula,
                     const char *domain, const char *units, const char *desc, float imp) {
    if (net->n_physics_laws >= 256) return;
    PhysicsLaw *l = &net->physics_laws[net->n_physics_laws++];
    strncpy(l->name,        name,    63);
    strncpy(l->formula,     formula, 127);
    strncpy(l->domain,      domain,  31);
    strncpy(l->units,       units,   63);
    strncpy(l->description, desc,    255);
    l->importance = imp;
}

void srhn4_load_physics(SRHNNetwork4 *net) {
    /* Classical Mechanics */
    add_law(net,"Newton's Second Law","F = m × a","mechanics","N = kg⋅m/s²",
            "Force equals mass times acceleration.",1.0f);
    add_law(net,"Newton's Law of Gravity","F = G × m₁ × m₂ / r²","mechanics","N",
            "Gravitational attraction between two masses.",1.0f);
    add_law(net,"Conservation of Energy","E_total = KE + PE = constant","mechanics","J",
            "Total energy of an isolated system is conserved.",1.0f);
    add_law(net,"Conservation of Momentum","p = m × v = constant","mechanics","kg⋅m/s",
            "Total momentum of an isolated system remains constant.",0.95f);
    add_law(net,"Hooke's Law","F = -k × x","mechanics","N/m",
            "Spring force is proportional to displacement.",0.85f);
    add_law(net,"Kepler's Third Law","T² ∝ a³","mechanics","s²/m³",
            "Orbital period squared proportional to semi-major axis cubed.",0.8f);
    add_law(net,"Centripetal Force","F = m × v² / r","mechanics","N",
            "Force required to keep an object moving in a circle.",0.85f);

    /* Thermodynamics */
    add_law(net,"First Law of Thermodynamics","ΔU = Q - W","thermodynamics","J",
            "Energy conservation: heat added minus work done equals change in internal energy.",1.0f);
    add_law(net,"Second Law of Thermodynamics","ΔS ≥ 0","thermodynamics","J/K",
            "Entropy of an isolated system never decreases.",1.0f);
    add_law(net,"Ideal Gas Law","PV = nRT","thermodynamics","Pa⋅m³",
            "Pressure-volume-temperature relation for ideal gases.",0.95f);
    add_law(net,"Stefan-Boltzmann Law","P = σ × A × T⁴","thermodynamics","W",
            "Total radiated power proportional to fourth power of temperature.",0.85f);
    add_law(net,"Carnot Efficiency","η = 1 - T_cold/T_hot","thermodynamics","dimensionless",
            "Maximum possible efficiency of a heat engine.",0.85f);

    /* Electromagnetism */
    add_law(net,"Coulomb's Law","F = k × q₁ × q₂ / r²","electromagnetism","N",
            "Electrostatic force between two charges.",1.0f);
    add_law(net,"Ohm's Law","V = I × R","electromagnetism","V = A⋅Ω",
            "Voltage equals current times resistance.",1.0f);
    add_law(net,"Faraday's Law","EMF = -dΦ/dt","electromagnetism","V",
            "Changing magnetic flux induces electromotive force.",0.95f);
    add_law(net,"Maxwell's Equations","∇⋅E=ρ/ε₀; ∇⋅B=0; ∇×E=-∂B/∂t; ∇×B=μ₀J+μ₀ε₀∂E/∂t",
            "electromagnetism","various","Complete description of classical electromagnetism.",1.0f);
    add_law(net,"Ampere's Law","∮B⋅dl = μ₀I","electromagnetism","T⋅m",
            "Magnetic field around a closed loop is proportional to enclosed current.",0.9f);

    /* Quantum */
    add_law(net,"Schrödinger Equation","iℏ ∂ψ/∂t = Ĥψ","quantum","J",
            "Governs quantum state evolution.",1.0f);
    add_law(net,"Heisenberg Uncertainty","Δx⋅Δp ≥ ℏ/2","quantum","kg⋅m²/s",
            "Position and momentum cannot both be known precisely.",1.0f);
    add_law(net,"Photoelectric Effect","E = hf - φ","quantum","J",
            "Energy of emitted electrons depends on photon frequency.",0.9f);
    add_law(net,"de Broglie Wavelength","λ = h / p","quantum","m",
            "Every particle has an associated wavelength.",0.85f);

    /* Relativity */
    add_law(net,"Mass-Energy Equivalence","E = mc²","relativity","J",
            "Mass and energy are equivalent and interconvertible.",1.0f);
    add_law(net,"Time Dilation","t' = t / √(1-v²/c²)","relativity","s",
            "Time passes slower for moving objects.",0.95f);
    add_law(net,"Length Contraction","L' = L × √(1-v²/c²)","relativity","m",
            "Moving objects are shorter in direction of motion.",0.9f);
    add_law(net,"Lorentz Factor","γ = 1 / √(1-v²/c²)","relativity","dimensionless",
            "Factor by which time, length, and mass change at relativistic speeds.",0.9f);

    /* Waves & Optics */
    add_law(net,"Wave Equation","v = f × λ","waves","m/s",
            "Wave velocity equals frequency times wavelength.",0.95f);
    add_law(net,"Snell's Law","n₁ sin θ₁ = n₂ sin θ₂","optics","dimensionless",
            "Light bends at interface between media.",0.9f);
    add_law(net,"Doppler Effect","f' = f × (v+vo)/(v-vs)","waves","Hz",
            "Observed frequency changes when source or observer moves.",0.85f);

    /* Build network nodes */
    for (uint32_t i = 0; i < net->n_physics_laws; i++) {
        PhysicsLaw *l = &net->physics_laws[i];
        char label[256];
        snprintf(label, 256, "%.60s: %.170s", l->name, l->description);
        uint32_t nid = srhn4_add_node(net, label, NODE_PHYSICS, LANG_ENGLISH);
        if (nid != UINT32_MAX) {
            strncpy(net->nodes[nid].formula, l->formula, 127);
            strncpy(net->nodes[nid].units,   l->units,   31);
            net->nodes[nid].importance = l->importance;
        }
    }

    /* Cross-connect related physics nodes (bounded: top-8 nearest per node) */
    uint32_t phys_start = (net->n_nodes > net->n_physics_laws)
                          ? net->n_nodes - net->n_physics_laws : 0;
    for (uint32_t i = phys_start; i < net->n_nodes; i++) {
        if (net->nodes[i].type != NODE_PHYSICS) continue;
        uint32_t nn_ids[8]; float nn_dists[8];
        int nn = srhn4_hnsw_search(&net->hnsw, &net->nodes[i].sig,
                                    16, nn_ids, nn_dists, 8, net);
        for (int k = 0; k < nn; k++) {
            if (nn_ids[k] == i) continue;
            if (net->nodes[nn_ids[k]].type != NODE_PHYSICS) continue;
            float r = 1.f - nn_dists[k];
            if (r > 0.35f)
                srhn4_connect(net, i, nn_ids[k], r * 0.8f, EDGE_ASSOC);
        }
    }
}

/* ── Code knowledge ──────────────────────────────────────────── */

void srhn4_load_code(SRHNNetwork4 *net) {
    /* Python */
    uint32_t py  = srhn4_add_node(net, "Python programming language", NODE_CODE, LANG_CODE);
    uint32_t pyf = srhn4_add_node(net, "Python function definition def", NODE_CODE, LANG_CODE);
    uint32_t pyc = srhn4_add_node(net, "Python class definition OOP", NODE_CODE, LANG_CODE);
    uint32_t pyl = srhn4_add_node(net, "Python list data structure", NODE_CODE, LANG_CODE);
    uint32_t pyd = srhn4_add_node(net, "Python dictionary key value", NODE_CODE, LANG_CODE);
    uint32_t pyi = srhn4_add_node(net, "Python import module package", NODE_CODE, LANG_CODE);
    uint32_t pya = srhn4_add_node(net, "Python async await coroutine", NODE_CODE, LANG_CODE);
    uint32_t pye = srhn4_add_node(net, "Python exception handling try except", NODE_CODE, LANG_CODE);
    if (py != UINT32_MAX) {
        strncpy(net->nodes[py].code_lang, "python", 15);
        srhn4_connect(net, py, pyf, 0.9f, EDGE_CODE);
        srhn4_connect(net, py, pyc, 0.9f, EDGE_CODE);
        srhn4_connect(net, py, pyl, 0.8f, EDGE_CODE);
        srhn4_connect(net, py, pyd, 0.8f, EDGE_CODE);
        srhn4_connect(net, py, pyi, 0.85f, EDGE_CODE);
        srhn4_connect(net, py, pya, 0.75f, EDGE_CODE);
        srhn4_connect(net, py, pye, 0.80f, EDGE_CODE);
    }

    /* C/C++ */
    uint32_t c   = srhn4_add_node(net, "C programming language", NODE_CODE, LANG_CODE);
    uint32_t cf  = srhn4_add_node(net, "C function void return type", NODE_CODE, LANG_CODE);
    uint32_t cs  = srhn4_add_node(net, "C struct data type aggregate", NODE_CODE, LANG_CODE);
    uint32_t cm  = srhn4_add_node(net, "C malloc free memory allocation", NODE_CODE, LANG_CODE);
    uint32_t cp  = srhn4_add_node(net, "C pointer arithmetic dereference", NODE_CODE, LANG_CODE);
    if (c != UINT32_MAX) {
        strncpy(net->nodes[c].code_lang, "c", 15);
        srhn4_connect(net, c, cf, 0.9f, EDGE_CODE);
        srhn4_connect(net, c, cs, 0.9f, EDGE_CODE);
        srhn4_connect(net, c, cm, 0.85f, EDGE_CODE);
        srhn4_connect(net, c, cp, 0.85f, EDGE_CODE);
    }

    /* JavaScript/TypeScript */
    uint32_t js  = srhn4_add_node(net, "JavaScript programming language", NODE_CODE, LANG_CODE);
    uint32_t jsf = srhn4_add_node(net, "JavaScript function arrow callback", NODE_CODE, LANG_CODE);
    uint32_t jsa = srhn4_add_node(net, "JavaScript async await Promise", NODE_CODE, LANG_CODE);
    uint32_t jsc = srhn4_add_node(net, "JavaScript class prototype inheritance", NODE_CODE, LANG_CODE);
    if (js != UINT32_MAX) {
        strncpy(net->nodes[js].code_lang, "javascript", 15);
        srhn4_connect(net, js, jsf, 0.9f, EDGE_CODE);
        srhn4_connect(net, js, jsa, 0.85f, EDGE_CODE);
        srhn4_connect(net, js, jsc, 0.80f, EDGE_CODE);
    }

    /* Rust */
    uint32_t rs  = srhn4_add_node(net, "Rust programming language memory safety", NODE_CODE, LANG_CODE);
    uint32_t rsb = srhn4_add_node(net, "Rust borrow checker ownership lifetime", NODE_CODE, LANG_CODE);
    uint32_t rse = srhn4_add_node(net, "Rust enum Result Option error handling", NODE_CODE, LANG_CODE);
    if (rs != UINT32_MAX) {
        strncpy(net->nodes[rs].code_lang, "rust", 15);
        srhn4_connect(net, rs, rsb, 0.95f, EDGE_CODE);
        srhn4_connect(net, rs, rse, 0.85f, EDGE_CODE);
    }

    /* Cross-language concepts */
    uint32_t func  = srhn4_add_concept(net, "function subroutine reusable code");
    uint32_t oop   = srhn4_add_concept(net, "object oriented programming class inheritance");
    uint32_t loop  = srhn4_add_concept(net, "loop iteration for while control flow");
    uint32_t recur = srhn4_add_concept(net, "recursion recursive base case");
    uint32_t async_c = srhn4_add_concept(net, "asynchronous concurrent parallel programming");
    uint32_t mem   = srhn4_add_concept(net, "memory management allocation deallocation");
    uint32_t types = srhn4_add_concept(net, "type system static dynamic typing");

    if (pyf != UINT32_MAX && func != UINT32_MAX) srhn4_connect(net, pyf, func, 0.9f, EDGE_CODE);
    if (cf  != UINT32_MAX && func != UINT32_MAX) srhn4_connect(net, cf,  func, 0.9f, EDGE_CODE);
    if (jsf != UINT32_MAX && func != UINT32_MAX) srhn4_connect(net, jsf, func, 0.9f, EDGE_CODE);
    if (pyc != UINT32_MAX && oop  != UINT32_MAX) srhn4_connect(net, pyc, oop,  0.9f, EDGE_CODE);
    if (jsc != UINT32_MAX && oop  != UINT32_MAX) srhn4_connect(net, jsc, oop,  0.9f, EDGE_CODE);
    if (py  != UINT32_MAX && loop != UINT32_MAX) srhn4_connect(net, py,  loop, 0.75f, EDGE_CODE);
    if (c   != UINT32_MAX && loop != UINT32_MAX) srhn4_connect(net, c,   loop, 0.75f, EDGE_CODE);
    if (pya != UINT32_MAX && async_c != UINT32_MAX) srhn4_connect(net, pya, async_c, 0.9f, EDGE_CODE);
    if (jsa != UINT32_MAX && async_c != UINT32_MAX) srhn4_connect(net, jsa, async_c, 0.9f, EDGE_CODE);
    if (cm  != UINT32_MAX && mem  != UINT32_MAX) srhn4_connect(net, cm,  mem,  0.9f, EDGE_CODE);
    if (rsb != UINT32_MAX && mem  != UINT32_MAX) srhn4_connect(net, rsb, mem,  0.9f, EDGE_CODE);

    /* Suppress unused warnings */
    (void)recur; (void)types;
}
