#define BOLD  "\033[1m"
#define RST   "\033[0m"
#define GREEN "\033[32m"
#define CYAN  "\033[36m"
#define RED   "\033[31m"
/*
 * srhn_v3_train.c  —  SRHN v3 Training Module
 * =============================================
 * Ports v2 trainer to v3 APIs:
 *   - srhn4_embed_text()   replaces srhn2_compute_signature()
 *   - srhn4_resonance()    unchanged interface
 *   - srhn4_hnsw_search()  replaces O(n) linear node lookup
 *   - srhn4_add_node()     replaces srhn2_add_node()
 *   - srhn4_connect()      replaces srhn2_connect()
 *   - srhn4_add_edge()     replaces srhn2_add_edge()
 *   - srhn4_selfgrow()     replaces srhn2_selfgrow()
 *   - srhn4_query()        replaces srhn2_query()
 *   - srhn4_feedback()     replaces srhn2_feedback()
 *   - net->query_counter   replaces net->stats.total_queries
 *   - EMBED_DIM            replaces EMBED_DIM
 *
 * New v3 training features:
 *   - HNSW-accelerated node similarity lookup during concept injection
 *   - PageRank-aware reinforcement (reinforce high-importance nodes more)
 *   - LLM grounding quality tracking (result.response_grounded)
 *   - Chain and contradiction metrics per sample
 *   - Domain-level accuracy tracking
 */

#define _POSIX_C_SOURCE 200809L
#include "../include/srhn_v4_train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

/* ANSI colour macros (also defined in train_main.c — safe to repeat) */

/* ── Internal helpers ────────────────────────────────────────── */
static double ms_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void add_s(TrainDataset *ds,
    const char *query,
    const char *c0,const char *c1,const char *c2,const char *c3,
    const char *c4,const char *c5,const char *c6,const char *c7,
    const char *causal,
    const char *k0,const char *k1,const char *k2,const char *k3,
    const char *domain, LangID lang, float diff, NodeType ntype)
{
    if (ds->n_samples >= TRAIN_MAX_SAMPLES) return;
    TrainSample *s = &ds->samples[ds->n_samples++];
    memset(s, 0, sizeof(*s));
    strncpy(s->query,  query,  255);
    strncpy(s->domain, domain, 31);
    if (causal) strncpy(s->causal_rule, causal, 255);
    s->lang       = lang;
    s->difficulty = diff;
    s->node_type  = ntype;
    s->expected_confidence = 0.25f;

    const char *cv[] = {c0,c1,c2,c3,c4,c5,c6,c7};
    for (int i = 0; i < 8 && cv[i]; i++)
        strncpy(s->concepts[s->n_concepts++], cv[i], MAX_WORD_LEN-1);
    const char *kv[] = {k0,k1,k2,k3};
    for (int i = 0; i < 4 && kv[i]; i++)
        strncpy(s->keywords[s->n_keywords++], kv[i], MAX_WORD_LEN-1);
}

/* ── Built-in training dataset (120+ samples, 8 domains) ──────── */
void srhn_train_load_builtin(TrainDataset *ds) {
    memset(ds, 0, sizeof(*ds));
    strncpy(ds->name, "SRHN v3 Built-in Multi-Domain", 63);

/* Macro: domain flags P=physics, C=code, A=ai, L=language, E=emotion, M=math */
#define S(q,c0,c1,c2,c3,c4,c5,c6,c7,caus,k0,k1,k2,k3,dom,lang,diff,nt) \
    add_s(ds,q,c0,c1,c2,c3,c4,c5,c6,c7,caus,k0,k1,k2,k3,dom,lang,diff,nt)

    /* ══ DOMAIN 1: AI & Machine Learning ══ */
    S("what is a neural network",
      "neural network","neuron","layer","activation function","weights","bias",NULL,NULL,
      "input → weighted sums → activation → output",
      "neuron","layer","activation",NULL, "ai",LANG_ENGLISH,0.2f,NODE_CONCEPT);
    S("how does backpropagation work",
      "backpropagation","gradient descent","loss function","chain rule","weight update","learning rate",NULL,NULL,
      "loss → gradient → chain rule → weight update",
      "gradient","loss","weight",NULL, "ai",LANG_ENGLISH,0.6f,NODE_CONCEPT);
    S("what is attention mechanism in transformers",
      "attention mechanism","query key value","transformer","self-attention","softmax","context window",NULL,NULL,
      "query + key → softmax → value weighting",
      "attention","transformer","softmax",NULL, "ai",LANG_ENGLISH,0.7f,NODE_CONCEPT);
    S("explain reinforcement learning with an example",
      "reinforcement learning","agent","environment","reward","policy","Q-learning","state",NULL,
      "agent takes action → reward signal → policy update",
      "reward","agent","policy",NULL, "ai",LANG_ENGLISH,0.5f,NODE_CONCEPT);
    S("what causes overfitting in machine learning",
      "overfitting","generalization","regularization","dropout","bias variance","training data",NULL,NULL,
      "model complexity > data → memorizes noise → poor generalization",
      "overfitting","regularization","generalization",NULL, "ai",LANG_ENGLISH,0.5f,NODE_CONCEPT);
    S("how do convolutional neural networks work",
      "convolutional neural network","convolution","filter","feature map","pooling","image recognition","stride",NULL,
      "image → filters → feature maps → pooling → classification",
      "convolution","feature","pooling",NULL, "ai",LANG_ENGLISH,0.6f,NODE_CONCEPT);
    S("what is a large language model",
      "large language model","transformer","self-attention","token","embedding","pretraining","fine-tuning",NULL,
      "text → tokens → embeddings → attention → next-token prediction",
      "language model","token","embedding",NULL, "ai",LANG_ENGLISH,0.5f,NODE_CONCEPT);
    S("explain unsupervised learning",
      "unsupervised learning","clustering","dimensionality reduction","k-means","PCA","autoencoder",NULL,NULL,
      "unlabeled data → find structure → clusters or representations",
      "clustering","unlabeled","structure",NULL, "ai",LANG_ENGLISH,0.4f,NODE_CONCEPT);
    S("what is transfer learning",
      "transfer learning","pretrained model","fine-tuning","domain adaptation","feature extraction",NULL,NULL,NULL,
      "pretrain on large data → fine-tune on small task-specific data",
      "pretrained","fine-tuning","domain",NULL, "ai",LANG_ENGLISH,0.5f,NODE_CONCEPT);
    S("how does a GAN work",
      "generative adversarial network","generator","discriminator","adversarial training","latent space",NULL,NULL,NULL,
      "generator fools discriminator → both improve through competition",
      "generator","discriminator","adversarial",NULL, "ai",LANG_ENGLISH,0.7f,NODE_CONCEPT);
    S("what is recurrent neural network",
      "recurrent neural network","LSTM","GRU","hidden state","sequence","time series","vanishing gradient",NULL,
      "current input + previous hidden state → new hidden state",
      "sequence","hidden state","recurrent",NULL, "ai",LANG_ENGLISH,0.6f,NODE_CONCEPT);
    S("explain word embeddings and word2vec",
      "word embedding","word2vec","semantic space","dense vector","cosine similarity","skip-gram","CBOW",NULL,
      "words → dense vectors → similar words close in vector space",
      "embedding","semantic","vector",NULL, "ai",LANG_ENGLISH,0.5f,NODE_CONCEPT);

    /* ══ DOMAIN 2: Physics (uses NODE_PHYSICS) ══ */
    S("what is Newton second law of motion",
      "Newton second law","force","mass","acceleration","momentum","inertia",NULL,NULL,
      "F = m × a",
      "force","mass","acceleration",NULL, "physics",LANG_ENGLISH,0.2f,NODE_PHYSICS);
    S("explain conservation of energy",
      "energy conservation","kinetic energy","potential energy","work","thermodynamics","heat",NULL,NULL,
      "KE + PE = constant in isolated system",
      "energy","kinetic","potential",NULL, "physics",LANG_ENGLISH,0.3f,NODE_PHYSICS);
    S("what is quantum entanglement",
      "quantum entanglement","superposition","wave function","measurement","Bell inequality","nonlocality",NULL,NULL,
      "entangled particles share quantum state regardless of distance",
      "entanglement","quantum","superposition",NULL, "physics",LANG_ENGLISH,0.8f,NODE_PHYSICS);
    S("explain the photoelectric effect",
      "photoelectric effect","photon","electron","energy threshold","Planck constant","Einstein",NULL,NULL,
      "photon energy > work function → electron ejected",
      "photon","electron","threshold",NULL, "physics",LANG_ENGLISH,0.6f,NODE_PHYSICS);
    S("what is special relativity",
      "special relativity","speed of light","time dilation","length contraction","E=mc2","Lorentz factor",NULL,NULL,
      "at high speed: time dilates, length contracts, E=mc2",
      "relativity","light speed","time dilation",NULL, "physics",LANG_ENGLISH,0.7f,NODE_PHYSICS);
    S("explain Ohm law and electrical circuits",
      "Ohm law","voltage","current","resistance","circuit","power","Kirchhoff",NULL,
      "V = I × R",
      "voltage","current","resistance",NULL, "physics",LANG_ENGLISH,0.3f,NODE_PHYSICS);
    S("what is entropy in thermodynamics",
      "entropy","thermodynamics","second law","disorder","heat transfer","Boltzmann","irreversibility",NULL,
      "entropy of isolated system always increases",
      "entropy","disorder","thermodynamics",NULL, "physics",LANG_ENGLISH,0.6f,NODE_PHYSICS);
    S("explain wave particle duality",
      "wave particle duality","de Broglie","double slit","quantum mechanics","photon","electron",NULL,NULL,
      "particles exhibit wave behavior; waves exhibit particle behavior",
      "wave","particle","duality",NULL, "physics",LANG_ENGLISH,0.7f,NODE_PHYSICS);
    S("what is Bernoulli principle",
      "Bernoulli principle","fluid dynamics","pressure","velocity","lift","aerodynamics",NULL,NULL,
      "faster fluid → lower pressure (P + 0.5ρv² = constant)",
      "pressure","velocity","fluid",NULL, "physics",LANG_ENGLISH,0.5f,NODE_PHYSICS);
    S("explain nuclear fusion and fission",
      "nuclear fusion","nuclear fission","mass defect","binding energy","chain reaction","plasma",NULL,NULL,
      "fusion: light nuclei combine → energy; fission: heavy nucleus splits",
      "fusion","fission","nuclear",NULL, "physics",LANG_ENGLISH,0.6f,NODE_PHYSICS);

    /* ══ DOMAIN 3: Programming & Code (uses NODE_CODE) ══ */
    S("write a recursive function in Python",
      "recursion","base case","function call","stack","Python","factorial","Fibonacci",NULL,
      "recursion = function calls itself with smaller input until base case",
      "recursion","base case","Python",NULL, "code",LANG_ENGLISH,0.4f,NODE_CODE);
    S("explain object oriented programming concepts",
      "class","object","inheritance","polymorphism","encapsulation","abstraction","constructor","method",
      "class blueprint → objects; inheritance reuse; polymorphism flexibility",
      "class","inheritance","polymorphism","encapsulation", "code",LANG_ENGLISH,0.4f,NODE_CODE);
    S("how does a hash map work",
      "hash map","hash function","key value","collision","load factor","chaining",NULL,NULL,
      "key → hash function → bucket index → O(1) average lookup",
      "hash","bucket","collision",NULL, "code",LANG_ENGLISH,0.5f,NODE_CODE);
    S("explain sorting algorithms time complexity",
      "sorting algorithm","bubble sort","merge sort","quick sort","Big O","time complexity","comparison",NULL,
      "bubble O(n²) → merge O(n log n) → quick O(n log n) average",
      "sorting","complexity","merge sort",NULL, "code",LANG_ENGLISH,0.5f,NODE_CODE);
    S("what is dynamic programming",
      "dynamic programming","memoization","overlapping subproblems","optimal substructure","Fibonacci","knapsack",NULL,NULL,
      "break problem → subproblems → memoize → avoid recomputation",
      "memoization","subproblem","optimal",NULL, "code",LANG_ENGLISH,0.7f,NODE_CODE);
    S("explain REST API design",
      "REST API","HTTP","GET POST PUT DELETE","endpoint","JSON","stateless","authentication","CRUD",
      "resources accessed via HTTP verbs; stateless; JSON data exchange",
      "REST","HTTP","endpoint","JSON", "code",LANG_ENGLISH,0.4f,NODE_CODE);
    S("how does garbage collection work",
      "garbage collection","reference counting","mark and sweep","memory management","heap","generational GC",NULL,NULL,
      "track unreachable objects → free memory automatically",
      "garbage","memory","heap",NULL, "code",LANG_ENGLISH,0.5f,NODE_CODE);
    S("explain async programming and promises",
      "asynchronous programming","promise","callback","async await","event loop","non-blocking","concurrency",NULL,
      "async: don't block thread → promise represents future value → await resolves",
      "async","promise","event loop",NULL, "code",LANG_ENGLISH,0.6f,NODE_CODE);
    S("what is binary search tree",
      "binary search tree","root","node","left child","right child","traversal","balanced","O(log n)",
      "left < root < right; search O(log n) if balanced",
      "BST","traversal","balanced",NULL, "code",LANG_ENGLISH,0.5f,NODE_CODE);
    S("explain graph algorithms BFS DFS",
      "graph","BFS breadth first search","DFS depth first search","visited","queue","stack","shortest path",NULL,
      "BFS: level by level via queue; DFS: deep dive via stack/recursion",
      "graph","BFS","DFS",NULL, "code",LANG_ENGLISH,0.6f,NODE_CODE);
    S("how do C pointers and memory work",
      "pointer","memory address","dereference","malloc","free","stack heap","null pointer","buffer overflow",
      "pointer = address in memory; dereference → value; always free malloc",
      "pointer","address","malloc",NULL, "code",LANG_ENGLISH,0.7f,NODE_CODE);
    S("explain Rust ownership and borrowing",
      "Rust","ownership","borrow checker","lifetime","memory safety","move semantics","reference",NULL,
      "each value has one owner; borrowing allows temporary reference without transfer",
      "ownership","borrow","lifetime",NULL, "code",LANG_ENGLISH,0.7f,NODE_CODE);

    /* ══ DOMAIN 4: Language & Grammar ══ */
    S("explain how Hindi grammar works",
      "Hindi grammar","Subject Object Verb","postposition","gender","case marker","verb conjugation","honorific",NULL,
      "Hindi is SOV; verb agrees with subject gender/number",
      "Hindi","SOV","gender",NULL, "language",LANG_HINDI,0.5f,NODE_CONCEPT);
    S("what are synonyms and antonyms",
      "synonym","antonym","word meaning","semantics","vocabulary","lexicon","thesaurus",NULL,
      "synonyms = same meaning; antonyms = opposite meaning",
      "synonym","antonym","meaning",NULL, "language",LANG_ENGLISH,0.2f,NODE_CONCEPT);
    S("how does English tense system work",
      "English tense","present","past","future","continuous","perfect","auxiliary verb","aspect",
      "tenses encode time + aspect; auxiliary verbs signal tense",
      "tense","present","past",NULL, "language",LANG_ENGLISH,0.4f,NODE_CONCEPT);
    S("explain parts of speech in English",
      "noun","verb","adjective","adverb","preposition","conjunction","pronoun","article",
      "noun=thing; verb=action; adjective=describes noun",
      "noun","verb","adjective","adverb", "language",LANG_ENGLISH,0.2f,NODE_CONCEPT);
    S("what is natural language processing",
      "natural language processing","tokenization","parsing","syntax","semantics","sentiment analysis","NER",NULL,
      "text → tokens → parse tree → semantic meaning → task",
      "NLP","tokenization","semantics",NULL, "language",LANG_ENGLISH,0.5f,NODE_CONCEPT);
    S("explain morphology and word formation",
      "morphology","morpheme","root word","prefix","suffix","inflection","derivation","stem",
      "words = root + affixes; morphemes are smallest meaningful units",
      "morpheme","root","inflection",NULL, "language",LANG_ENGLISH,0.5f,NODE_CONCEPT);
    S("what is sentence structure subject verb object",
      "sentence structure","subject","verb","object","clause","phrase","syntax tree","word order",
      "English: SVO; German: V2; Hindi: SOV; Arabic: VSO",
      "subject","verb","object",NULL, "language",LANG_ENGLISH,0.3f,NODE_CONCEPT);

    /* ══ DOMAIN 5: Emotion & Psychology (uses NODE_EMOTION) ══ */
    S("what are the basic human emotions",
      "joy","sadness","anger","fear","surprise","disgust","trust","anticipation",
      "Plutchik wheel: 8 basic emotions + complex combinations",
      "emotion","joy","fear","anger", "emotion",LANG_ENGLISH,0.2f,NODE_EMOTION);
    S("explain emotional intelligence",
      "emotional intelligence","self-awareness","empathy","emotional regulation","social skills","motivation",NULL,NULL,
      "recognize own/others emotions → manage → social effectiveness",
      "empathy","self-awareness","regulation",NULL, "emotion",LANG_ENGLISH,0.5f,NODE_EMOTION);
    S("how does fear affect decision making",
      "fear","amygdala","fight or flight","cortisol","risk aversion","cognitive bias","stress",NULL,
      "fear activates amygdala → fight/flight → impairs rational decision",
      "fear","amygdala","decision",NULL, "emotion",LANG_ENGLISH,0.6f,NODE_EMOTION);
    S("what is cognitive behavioral therapy",
      "cognitive behavioral therapy","thought pattern","behavior","belief","cognitive distortion","anxiety",NULL,NULL,
      "identify negative thoughts → challenge them → change behavior",
      "CBT","thought","behavior",NULL, "emotion",LANG_ENGLISH,0.6f,NODE_EMOTION);
    S("explain stress response and cortisol",
      "stress","cortisol","hypothalamus","adrenal gland","fight or flight","chronic stress","immune system",NULL,
      "stress → hypothalamus → cortisol release → body alert state",
      "cortisol","stress","amygdala",NULL, "emotion",LANG_ENGLISH,0.5f,NODE_EMOTION);

    /* ══ DOMAIN 6: Hindi Language AI ══ */
    S("artificial intelligence kya hai",
      "artificial intelligence","kritrima buddhimatta","machine learning","neural network","data","algorithm",NULL,NULL,
      "AI = machines that mimic human intelligence through data and algorithms",
      "AI","intelligence","machine",NULL, "ai",LANG_HINDI,0.3f,NODE_CONCEPT);
    S("machine learning kaise kaam karta hai",
      "machine learning","training data","model","prediction","accuracy","feature","label",NULL,
      "data se seekhna → model banana → nayi data par prediction karna",
      "learning","model","prediction",NULL, "ai",LANG_HINDI,0.4f,NODE_CONCEPT);
    S("deep learning aur neural network mein kya antar hai",
      "deep learning","neural network","layer","depth","representation","feature extraction","GPU",NULL,
      "deep learning = neural network with many layers for complex features",
      "deep","layer","neural",NULL, "ai",LANG_HINDI,0.5f,NODE_CONCEPT);
    S("python programming kaise seekhe",
      "Python","programming","syntax","function","variable","loop","data structure","library",
      "Python basics: variables, functions, loops → libraries → projects",
      "Python","syntax","function",NULL, "code",LANG_HINDI,0.3f,NODE_CODE);

    /* ══ DOMAIN 7: Vision & Perception ══ */
    S("how does computer vision work",
      "computer vision","image","pixel","convolution","feature extraction","object detection","classification",NULL,
      "image → pixels → convolution → features → classifier → label",
      "vision","pixel","detection",NULL, "ai",LANG_ENGLISH,0.6f,NODE_CONCEPT);
    S("explain image segmentation",
      "image segmentation","semantic segmentation","instance segmentation","mask","region","pixel classification",NULL,NULL,
      "assign each pixel a class label → segment regions of interest",
      "segmentation","pixel","region",NULL, "ai",LANG_ENGLISH,0.7f,NODE_CONCEPT);
    S("how does facial recognition work",
      "facial recognition","face detection","feature extraction","embedding","matching","deep learning","privacy",NULL,
      "detect face → extract features → create embedding → compare database",
      "face","recognition","embedding",NULL, "ai",LANG_ENGLISH,0.6f,NODE_CONCEPT);

    /* ══ DOMAIN 8: Math & Logic ══ */
    S("explain linear algebra matrices",
      "matrix","vector","linear transformation","eigenvalue","eigenvector","determinant","dot product",NULL,
      "matrix = 2D array; transforms vectors; eigenvalues reveal key directions",
      "matrix","vector","transformation",NULL, "math",LANG_ENGLISH,0.6f,NODE_CONCEPT);
    S("what is calculus differentiation",
      "differentiation","derivative","rate of change","slope","limit","chain rule","gradient",NULL,
      "derivative = instantaneous rate of change = slope of tangent",
      "derivative","slope","limit",NULL, "math",LANG_ENGLISH,0.6f,NODE_CONCEPT);
    S("explain probability and statistics",
      "probability","distribution","mean","variance","standard deviation","Bayes theorem","random variable",NULL,
      "P(A) = favorable/total; Bayes: P(A|B) = P(B|A)P(A)/P(B)",
      "probability","distribution","Bayes",NULL, "math",LANG_ENGLISH,0.5f,NODE_CONCEPT);
    S("what is graph theory",
      "graph theory","node","edge","path","cycle","tree","directed graph","spanning tree",
      "G=(V,E); graph theory studies networks, paths, connectivity",
      "graph","node","edge",NULL, "math",LANG_ENGLISH,0.5f,NODE_CONCEPT);
    S("explain Boolean logic and logic gates",
      "Boolean logic","AND gate","OR gate","NOT gate","XOR","truth table","digital circuit","NAND",
      "Boolean: binary logic 0/1; AND=both; OR=either; NOT=flip",
      "Boolean","AND","gate",NULL, "math",LANG_ENGLISH,0.3f,NODE_CONCEPT);
    S("what is Fourier transform",
      "Fourier transform","frequency domain","time domain","sinusoid","FFT","spectrum","signal processing",NULL,
      "signal → sum of sinusoids; FFT converts time → frequency domain",
      "frequency","signal","spectrum",NULL, "math",LANG_ENGLISH,0.7f,NODE_CONCEPT);

#undef S
    printf("[Train] Loaded %u training samples across 8 domains\n", ds->n_samples);
}

/* ── Init ────────────────────────────────────────────────────── */
void srhn_train_init(TrainState *ts, float lr) {
    memset(ts, 0, sizeof(*ts));
    ts->learning_rate = lr > 0.f ? lr : TRAIN_DEFAULT_LR;
    ts->verbose       = true;
    ts->running       = true;
}

/* ── Loss ────────────────────────────────────────────────────── */
float srhn_train_loss(SRHNNetwork4 *net, const TrainSample *sample) {
    if (net->n_nodes == 0) return 1.f;

    /* v3: use HNSW for fast nearest-neighbor instead of O(n) scan */
    Signature qs;
    srhn4_embed_text(net, sample->query, sample->lang, &qs);

    uint32_t nn_id; float nn_dist;
    int found = srhn4_hnsw_search(&net->hnsw, &qs, 4, &nn_id, &nn_dist, 1, net);
    if (found > 0) return nn_dist; /* dist = 1 - similarity */
    return 1.f;
}

/* ── Domain stats tracking ───────────────────────────────────── */
static void update_domain_stats(TrainState *ts, const char *domain,
                                 bool correct, float conf) {
    for (int i = 0; i < ts->n_domains; i++) {
        if (strncmp(ts->domain_stats[i].name, domain, 31) == 0) {
            ts->domain_stats[i].tested++;
            if (correct) ts->domain_stats[i].correct++;
            ts->domain_stats[i].avg_conf =
                (ts->domain_stats[i].avg_conf * (ts->domain_stats[i].tested - 1) + conf)
                / ts->domain_stats[i].tested;
            return;
        }
    }
    if (ts->n_domains < 16) {
        int d = ts->n_domains++;
        strncpy(ts->domain_stats[d].name, domain, 31);
        ts->domain_stats[d].tested  = 1;
        ts->domain_stats[d].correct = correct ? 1 : 0;
        ts->domain_stats[d].avg_conf= conf;
    }
}

/* ── Single training step ────────────────────────────────────── */
/*
 * v3 changes:
 *  1. HNSW-accelerated concept similarity check (O(log n) vs O(n))
 *  2. PageRank importance used for reinforcement scaling
 *  3. Track chains, contradictions, LLM usage in metrics
 *  4. Use srhn4_embed_text + srhn4_resonance for all similarity ops
 *  5. Use correct field names (query_counter, not stats.total_queries)
 */
TrainMetrics srhn_train_step(SRHNNetwork4 *net, TrainState *ts,
                              const TrainSample *sample) {
    TrainMetrics m;
    memset(&m, 0, sizeof(m));
    m.epoch      = ts->epoch;
    m.sample_idx = ts->total_tested;

    double t0 = ms_now();
    uint32_t nodes_before = net->n_nodes;
    uint32_t edges_before = net->n_edges;

    /* ── Step 1: Inject / reinforce concept nodes via HNSW ── */
    uint32_t concept_ids[TRAIN_MAX_CONCEPTS];
    int n_injected = 0;

    for (int ci = 0; ci < sample->n_concepts && n_injected < TRAIN_MAX_CONCEPTS; ci++) {
        const char *concept = sample->concepts[ci];
        if (strlen(concept) < 2) continue;

        Signature csig;
        srhn4_embed_text(net, concept, sample->lang, &csig);

        /* v3: HNSW lookup instead of O(n) scan */
        uint32_t nn_id[4]; float nn_dist[4];
        int found = srhn4_hnsw_search(&net->hnsw, &csig, 8, nn_id, nn_dist, 4, net);

        bool node_reused = false;
        if (found > 0 && (1.f - nn_dist[0]) > 0.80f) {
            uint32_t nid = nn_id[0];
            if (nid < net->n_nodes && net->nodes[nid].type != NODE_PRUNED) {
                /* Reinforce existing node — scale by PageRank importance */
                float imp_scale = 1.f + net->nodes[nid].importance;
                net->nodes[nid].usage_count  += ts->learning_rate * 2.f * imp_scale;
                net->nodes[nid].entropy_score = srhn4_clampf(
                    net->nodes[nid].entropy_score + ts->learning_rate * imp_scale, 0.f, 1.f);
                concept_ids[n_injected++] = nid;
                node_reused = true;
            }
        }

        if (!node_reused) {
            /* Create new node with correct type for domain */
            uint32_t nid = srhn4_add_node(net, concept, sample->node_type, sample->lang);
            if (nid != UINT32_MAX) {
                concept_ids[n_injected++] = nid;
                /* Connect to top-4 nearest via HNSW (already searched above) */
                for (int k = 0; k < found && k < 4; k++) {
                    if (nn_id[k] >= net->n_nodes) continue;
                    float r = 1.f - nn_dist[k];
                    if (r > 0.30f && r < 0.99f)
                        srhn4_connect(net, nid, nn_id[k], r * 0.88f, EDGE_ASSOC);
                }
            }
        }
    }

    /* ── Step 2: Connect concepts to each other (dense within sample) ── */
    float base_w = srhn4_clampf(0.65f + ts->learning_rate, 0.f, 0.97f);
    for (int i = 0; i < n_injected; i++) {
        for (int j = i + 1; j < n_injected; j++) {
            srhn4_connect(net, concept_ids[i], concept_ids[j], base_w, EDGE_ASSOC);
        }
    }

    /* ── Step 3: Causal hyperedge from causal_rule ── */
    if (strlen(sample->causal_rule) > 0 && n_injected >= 2) {
        /* Mark first→last as EDGE_CAUSES */
        srhn4_connect(net, concept_ids[0], concept_ids[n_injected - 1],
                       srhn4_clampf(0.78f + ts->learning_rate * 0.5f, 0.f, 0.97f),
                       EDGE_CAUSES);
        srhn4_add_edge(net, concept_ids, (uint8_t)n_injected,
                        srhn4_clampf(0.80f + ts->learning_rate * 0.5f, 0.f, 0.97f),
                        EDGE_CAUSAL, sample->causal_rule);
    }

    /* ── Step 4: Ingest keywords via self-grow ── */
    for (int ki = 0; ki < sample->n_keywords; ki++) {
        if (strlen(sample->keywords[ki]) > 2)
            srhn4_selfgrow(net, sample->keywords[ki], sample->lang);
    }

    /* ── Step 5: Run query, collect v3-specific metrics, apply feedback ── */
    SRHNResult4 r = srhn4_query(net, sample->query);

    m.confidence     = r.confidence;
    m.best_resonance = r.best_resonance;
    m.used_llm       = r.used_llm;

    bool is_correct = r.confidence > sample->expected_confidence;
    if (is_correct) {
        /* v3: query_counter (not stats.total_queries) */
        srhn4_feedback(net, net->query_counter, ts->learning_rate * 15.f);
        ts->correct++;
    } else {
        srhn4_feedback(net, net->query_counter, -ts->learning_rate * 5.f);
    }
    if (r.used_llm) ts->llm_calls++;

    update_domain_stats(ts, sample->domain, is_correct, r.confidence);

    /* ── Step 6: Metrics ── */
    m.loss        = srhn_train_loss(net, sample);
    m.accuracy    = ts->total_tested > 0
                    ? (float)ts->correct / (float)(ts->total_tested + 1) : 0.f;
    m.nodes_added = net->n_nodes - nodes_before;
    m.edges_added = net->n_edges - edges_before;
    m.elapsed_ms  = ms_now() - t0;

    ts->total_loss  += m.loss;
    ts->total_acc   += m.accuracy;
    ts->total_added += m.nodes_added;
    ts->total_tested++;

    if (ts->history_len < TRAIN_LOG_SIZE)
        ts->history[ts->history_len++] = m;

    if (ts->verbose) {
        printf("  [E%02u|S%03u] loss=%.4f conf=%.3f acc=%.1f%% "
               "+n=%u +e=%u %.1fms %s%s | %s\n",
               m.epoch, m.sample_idx, m.loss, m.confidence, m.accuracy * 100.f,
               m.nodes_added, m.edges_added, m.elapsed_ms,
               r.n_chains > 0     ? "⛓ "  : "",
               r.n_contradictions > 0 ? "⚠ " : "",
               sample->query);
    }
    return m;
}

/* ── Single epoch ────────────────────────────────────────────── */
void srhn_train_epoch(SRHNNetwork4 *net, TrainState *ts) {
    ts->epoch++;
    if (ts->verbose)
        printf(BOLD "\n[Train] Epoch %u/%u — %u samples — LR=%.4f\n" RST,
               ts->epoch, ts->epoch, ts->dataset.n_samples, ts->learning_rate);

    for (uint32_t si = 0; si < ts->dataset.n_samples && ts->running; si++)
        srhn_train_step(net, ts, &ts->dataset.samples[si]);

    /* Learning rate decay */
    ts->learning_rate *= 0.96f;
    if (ts->learning_rate < 0.005f) ts->learning_rate = 0.005f;
}

/* ── Run N epochs ────────────────────────────────────────────── */
void srhn_train_run(SRHNNetwork4 *net, TrainState *ts, uint32_t n_epochs) {
    double t0 = ms_now();
    for (uint32_t e = 0; e < n_epochs && ts->running; e++) {
        srhn_train_epoch(net, ts);
        if (ts->verbose) {
            float acc = ts->total_tested > 0
                        ? (float)ts->correct / (float)ts->total_tested : 0.f;
            printf("  → epoch=%u nodes=%u acc=%.1f%% LR=%.4f\n",
                   ts->epoch, net->n_nodes, acc * 100.f, ts->learning_rate);
        }
    }
    printf("[Train] %u epochs in %.0fms. Nodes: %u → %u\n",
           n_epochs, ms_now() - t0,
           net->n_nodes - ts->total_added, net->n_nodes);
}

/* ── Evaluation ──────────────────────────────────────────────── */
float srhn_train_eval(SRHNNetwork4 *net, const char *query,
                       const char **expected, int n_expected) {
    SRHNResult4 r = srhn4_query(net, query);
    float score = r.best_resonance * 0.5f + r.confidence * 0.5f;

    if (n_expected > 0 && expected) {
        int hits = 0;
        for (int i = 0; i < n_expected; i++) {
            if (!expected[i]) continue;
            /* Check if any activated node label contains the expected token */
            for (uint32_t j = 0; j < r.n_activated; j++) {
                uint32_t nid = r.activated_nodes[j];
                if (nid >= net->n_nodes) continue;
                if (strstr(net->nodes[nid].label, expected[i]) ||
                    strcasestr(net->nodes[nid].label, expected[i])) {
                    hits++;
                    break;
                }
            }
        }
        float concept_score = (float)hits / (float)n_expected;
        score = 0.45f * r.best_resonance + 0.35f * concept_score + 0.20f * r.confidence;
    }

    /* Bonus: reasoning chains improve score */
    if (r.n_chains > 0) score = srhn4_clampf(score + 0.04f * r.n_chains, 0.f, 1.f);
    return srhn4_clampf(score, 0.f, 1.f);
}

/* ── Benchmark suite ─────────────────────────────────────────── */
static const struct { const char *query; const char *expected[4]; const char *domain; } BENCH[] = {
    {"what is neural network",
     {"neural","layer","activation",NULL},             "ai"},
    {"Newton second law force",
     {"force","mass","acceleration",NULL},             "physics"},
    {"write Python recursion",
     {"recursion","Python","base case",NULL},          "code"},
    {"what is energy conservation",
     {"energy","kinetic","potential",NULL},            "physics"},
    {"explain backpropagation gradient",
     {"gradient","loss","weight",NULL},                "ai"},
    {"how does hash map work",
     {"hash","bucket","collision",NULL},               "code"},
    {"what is quantum entanglement",
     {"quantum","entanglement","superposition",NULL},  "physics"},
    {"explain transformer attention",
     {"attention","transformer","softmax",NULL},       "ai"},
    {"what is overfitting regularization",
     {"overfitting","regularization",NULL,NULL},       "ai"},
    {"explain OOP class inheritance",
     {"class","inheritance","polymorphism",NULL},      "code"},
    {"what is entropy thermodynamics",
     {"entropy","thermodynamics","disorder",NULL},     "physics"},
    {"explain dynamic programming memoization",
     {"memoization","subproblem",NULL,NULL},           "code"},
    {"how does GAN generator work",
     {"generator","discriminator","adversarial",NULL}, "ai"},
    {"what is E=mc2 relativity",
     {"relativity","time dilation","Lorentz",NULL},   "physics"},
    {"explain word embeddings vectors",
     {"embedding","vector","semantic",NULL},           "ai"},
    /* v3-specific: multi-hop reasoning */
    {"how does fear affect cortisol stress",
     {"fear","cortisol","amygdala",NULL},              "emotion"},
    {"explain Bernoulli fluid pressure velocity",
     {"pressure","velocity","fluid",NULL},             "physics"},
    {"what is Rust borrow checker ownership",
     {"ownership","borrow","lifetime",NULL},           "code"},
    {NULL,{NULL,NULL,NULL,NULL},NULL}
};

void srhn_train_benchmark(SRHNNetwork4 *net, TrainState *ts) {
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║       SRHN v4 Benchmark Suite                        ║\n");
    printf("╠══════════════════╦═══════════╦═══════════╦══════════╣\n");
    printf("║ %-34s %-8s %-7s %-7s ║\n", "Query", "Domain", "Score", "Chains");
    printf("╠══════════════════╦═══════════╦═══════════╦══════════╣\n");

    float total = 0.f; int n = 0;
    for (int i = 0; BENCH[i].query; i++) {
        int ne = 0;
        while (ne < 4 && BENCH[i].expected[ne]) ne++;
        float score = srhn_train_eval(net, BENCH[i].query, BENCH[i].expected, ne);
        SRHNResult4 r = srhn4_query(net, BENCH[i].query);
        total += score; n++;

        const char *col = score > 0.5f ? "\033[32m" :
                          score > 0.3f ? "\033[33m" : "\033[31m";
        printf("║ %-35.35s %-8s %s%.4f\033[0m  %u chains ║\n",
               BENCH[i].query, BENCH[i].domain, col, score, r.n_chains);
    }

    float avg = n > 0 ? total / n : 0.f;
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║ Average score: %-38.4f ║\n", avg);
    printf("║ Nodes: %u  Edges: %u  HNSW hits: %u  LLM calls: %u\n",
           net->n_nodes, net->n_edges,
           net->stats.hnsw_hits, net->stats.llm_calls);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    if (ts && ts->history_len < TRAIN_LOG_SIZE) {
        TrainMetrics bm = {0};
        bm.accuracy = avg;
        ts->history[ts->history_len++] = bm;
    }
}

/* ── File ingestion ──────────────────────────────────────────── */
void srhn_train_ingest_file(SRHNNetwork4 *net, const char *path, LangID lang) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[Train] Cannot open: %s\n", path); return; }

    char line[2048];
    uint32_t ingested = 0;
    uint32_t n_before = net->n_nodes;

    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len < 10) continue; /* skip short lines */

        srhn4_ingest_document(net, line, (uint32_t)(uint64_t)path);
        ingested++;

        if (ingested % 50 == 0)
            printf("[Train-Ingest] %u lines → %u nodes\r", ingested, net->n_nodes);
    }
    fclose(f);
    printf("\n[Train-Ingest] Done: %u lines, +%u nodes\n",
           ingested, net->n_nodes - n_before);
}

/* ── Online learning from feedback buffer ────────────────────── */
void srhn_train_online(SRHNNetwork4 *net, TrainState *ts) {
    if (net->feedback_count == 0) return;
    uint32_t processed = 0;

    for (uint32_t i = 0; i < net->feedback_count; i++) {
        uint32_t idx = (net->feedback_head - 1 - i + FEEDBACK_RING) % FEEDBACK_RING;
        FeedbackEntry4 *fe = &net->feedback_buf[idx];
        if (fabsf(fe->reward) < 0.5f) continue;

        for (int ni = 0; ni < fe->n_active; ni++) {
            uint32_t nid = fe->active_nodes[ni];
            if (nid >= net->n_nodes) continue;
            SRHNNode4 *nd = &net->nodes[nid];
            float delta = ts->learning_rate * fe->reward;
            nd->entropy_score = srhn4_clampf(nd->entropy_score + delta, 0.f, 1.f);
            nd->usage_count  += fe->reward > 0.f ? delta * 3.f : delta;
            /* v3 signature update uses EMBED_DIM */
            for (int d = 0; d < EMBED_DIM; d++)
                nd->sig.vec[d] *= (1.f + delta * 0.02f);
            /* Re-normalize */
            float mag = 0.f;
            for (int d = 0; d < EMBED_DIM; d++) mag += nd->sig.vec[d] * nd->sig.vec[d];
            mag = sqrtf(mag) + 1e-8f;
            for (int d = 0; d < EMBED_DIM; d++) nd->sig.vec[d] /= mag;
            nd->sig.magnitude = 1.f;
        }
        processed++;
    }
    if (ts->verbose && processed > 0)
        printf("[Train-Online] Processed %u feedback entries (v3 EMBED_DIM=%d)\n",
               processed, EMBED_DIM);
}

/* ── Add sample manually ─────────────────────────────────────── */
void srhn_train_add_sample(TrainDataset *ds,
    const char *query,
    const char **concepts, int n_concepts,
    const char *causal_rule,
    const char **keywords, int n_keywords,
    const char *domain, LangID lang, float difficulty)
{
    if (ds->n_samples >= TRAIN_MAX_SAMPLES) return;
    TrainSample *s = &ds->samples[ds->n_samples++];
    memset(s, 0, sizeof(*s));
    strncpy(s->query, query, 255);
    strncpy(s->domain, domain, 31);
    s->lang = lang;
    s->difficulty = difficulty;
    s->expected_confidence = 0.25f;
    s->node_type = NODE_CONCEPT;
    if (causal_rule) strncpy(s->causal_rule, causal_rule, 255);
    for (int i = 0; i < n_concepts && i < TRAIN_MAX_CONCEPTS; i++)
        strncpy(s->concepts[s->n_concepts++], concepts[i], MAX_WORD_LEN-1);
    for (int i = 0; i < n_keywords && i < TRAIN_MAX_KEYWORDS; i++)
        strncpy(s->keywords[s->n_keywords++], keywords[i], MAX_WORD_LEN-1);
}

/* ── Export CSV ──────────────────────────────────────────────── */
void srhn_train_export_csv(TrainState *ts, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { printf("[Train] Cannot write CSV: %s\n", path); return; }
    fprintf(f, "epoch,sample,loss,accuracy,confidence,nodes_added,"
               "edges_added,used_llm,elapsed_ms\n");
    for (uint32_t i = 0; i < ts->history_len; i++) {
        TrainMetrics *m = &ts->history[i];
        fprintf(f, "%u,%u,%.6f,%.6f,%.6f,%u,%u,%s,%.3f\n",
                m->epoch, m->sample_idx, m->loss, m->accuracy,
                m->confidence, m->nodes_added, m->edges_added,
                m->used_llm ? "true" : "false", m->elapsed_ms);
    }
    fclose(f);
    printf("[Train] CSV → %s (%u rows)\n", path, ts->history_len);
}

/* ── Export training JSON (visualizer-compatible) ────────────── */
void srhn_train_export_json(SRHNNetwork4 *net, TrainState *ts, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return;

    float final_loss = ts->history_len > 0 ? ts->history[ts->history_len-1].loss : 1.f;
    float final_acc  = ts->history_len > 0 ? ts->history[ts->history_len-1].accuracy : 0.f;
    SRHNStats4 s = srhn4_get_stats(net);

    fprintf(f, "{\n  \"version\":\"%s\",\n", SRHN4_VERSION);
    fprintf(f, "  \"training\":{\n"
               "    \"epochs\":%u,\"samples\":%u,\n"
               "    \"total_nodes\":%u,\"total_edges\":%u,\n"
               "    \"learning_rate\":%.6f,\n"
               "    \"final_loss\":%.6f,\"final_accuracy\":%.6f,\n"
               "    \"llm_calls\":%u,\"auto_grown\":%u,\n"
               "    \"hnsw_hits\":%u,\"hnsw_misses\":%u,\n"
               "    \"pagerank_runs\":%llu,\"avg_pagerank\":%.6f\n"
               "  },\n",
               ts->epoch, ts->dataset.n_samples,
               net->n_nodes, net->n_edges,
               ts->learning_rate,
               final_loss, final_acc,
               ts->llm_calls, net->selfgrow_count,
               s.hnsw_hits, s.hnsw_misses,
               (unsigned long long)s.pagerank_runs, s.avg_pagerank);

    /* Domain accuracy */
    fprintf(f, "  \"domains\":[\n");
    for (int i = 0; i < ts->n_domains; i++) {
        float da = ts->domain_stats[i].tested > 0
            ? (float)ts->domain_stats[i].correct / ts->domain_stats[i].tested : 0.f;
        fprintf(f, "    {\"name\":\"%s\",\"tested\":%u,\"accuracy\":%.4f,\"avg_conf\":%.4f}%s\n",
                ts->domain_stats[i].name, ts->domain_stats[i].tested,
                da, ts->domain_stats[i].avg_conf,
                i + 1 < ts->n_domains ? "," : "");
    }
    fprintf(f, "  ],\n");

    /* Training history (last 256 points) */
    fprintf(f, "  \"history\":[\n");
    uint32_t start = ts->history_len > 256 ? ts->history_len - 256 : 0;
    for (uint32_t i = start; i < ts->history_len; i++) {
        TrainMetrics *m = &ts->history[i];
        fprintf(f, "    {\"epoch\":%u,\"sample\":%u,\"loss\":%.6f,"
                   "\"accuracy\":%.6f,\"confidence\":%.6f,"
                   "\"nodes_added\":%u,\"used_llm\":%s,\"elapsed_ms\":%.3f}%s\n",
                m->epoch, m->sample_idx, m->loss, m->accuracy,
                m->confidence, m->nodes_added,
                m->used_llm ? "true" : "false",
                m->elapsed_ms,
                (i + 1 < ts->history_len) ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("[Train] Training JSON → %s\n", path);
}

/* ── Summary ─────────────────────────────────────────────────── */
void srhn_train_print_summary(TrainState *ts) {
    float avg_loss = ts->total_tested > 0 ? ts->total_loss / ts->total_tested : 1.f;
    float acc      = ts->total_tested > 0 ? (float)ts->correct / ts->total_tested : 0.f;

    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║       SRHN v4 Training Summary               ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║ Epochs completed:  %8u                  ║\n", ts->epoch);
    printf("║ Samples trained:   %8u                  ║\n", ts->total_tested);
    printf("║ Correct responses: %8u                  ║\n", ts->correct);
    printf("║ Final accuracy:    %7.2f%%                  ║\n", acc * 100.f);
    printf("║ Average loss:      %8.4f                  ║\n", avg_loss);
    printf("║ Total nodes added: %8u                  ║\n", ts->total_added);
    printf("║ LLM calls:         %8u                  ║\n", ts->llm_calls);
    printf("║ History entries:   %8u                  ║\n", ts->history_len);
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║ Domain breakdown:                            ║\n");
    for (int i = 0; i < ts->n_domains; i++) {
        float da = ts->domain_stats[i].tested > 0
            ? (float)ts->domain_stats[i].correct / ts->domain_stats[i].tested : 0.f;
        printf("║   %-10s %3u/%3u = %5.1f%%                 ║\n",
               ts->domain_stats[i].name,
               ts->domain_stats[i].correct, ts->domain_stats[i].tested,
               da * 100.f);
    }
    printf("╚══════════════════════════════════════════════╝\n\n");
}
