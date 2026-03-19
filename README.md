# Morphosyntactic Fingerprinting of Editorial Layers in the Greek New Testament

A grammar-only deep learning approach for detecting editorial layers in ancient Greek texts. The classifier operates exclusively on morphosyntactic features, PROIEL morphological codes, dependency relations, and sentence-level grammar profiles, with **zero lexical access**. It cannot see words, only grammatical structure.

## Key Results

- **AUC-ROC 0.892** (96.3% accuracy on confident predictions at ±0.30 margin) distinguishing Jewish-tradition from Gentile-tradition Greek using grammar alone
- **21 robust J→G tradition flips** across three independent marker sets, validated by random ablation (0/100 random flips each)
  - Original (Christological/patriarchal): 12 robust, combined p < 8.87 × 10⁻²⁵
  - Catholic (ecclesiological/creedal/pneumatological): 3 robust, combined p < 9.71 × 10⁻⁷
  - Expanded (θεός/πατήρ/πίστις/σωτηρία): 6 robust, combined p < 9.42 × 10⁻¹³
- Reverse direction (G→J) tested across all three sets: **2 robust flips** (10.5:1 asymmetry)
- Several robust flips resolve known exegetical difficulties (Heb 6, Phil 2, Phil 3:8, 1 Pet 1)

## Architecture

**MorphTagTransformer** (358K parameters):

```
Per token: [62d PROIEL morph one-hot | 16d deprel embedding] = 78d
    → Linear(78, 128) projection
    → + Learned positional encoding
    → TransformerEncoder (2 layers, 4 heads, d_model=128, pre-norm, GELU)
    → Mean pool (masked)
    → Concat with sentence grammar profile (391d)
    → MLP(519 → 128 → 64 → 2) classifier head
```

**Deep Ensemble**: 10 models with different random seeds (42-51). Averaged softmax for prediction, standard deviation for uncertainty quantification.

## Repository Structure

```
morphosyntactic_fingerprinting/
├── README.md
├── requirements.txt
├── LICENSE
├── models/
│   └── morph_transformer.py          # MorphTagTransformer architecture
├── trained_models/
│   └── grammar_ensemble_seed{42-51}.pt  # 10 pre-trained model weights
├── Corpus/
│   └── training_corpus.tar.gz        # Compressed training data (jewish/ + gentile/ CoNLL-U)
├── new-testament/                    # NT CoNLL-U files (27 books, MorphGNT + PROIEL format)
├── data/
│   ├── conllu_parser.py              # CoNLL-U file parser
│   ├── morphology_normalizer.py      # PROIEL ↔ CoNLL-U conversion
│   ├── feature_extractor.py          # 391-dim grammar profile extraction
│   └── constants.py                  # PROIEL vocabularies + model dimensions
├── analysis/
│   ├── train_ensemble.py             # Train 10-model deep ensemble
│   ├── soft_accuracy_analysis.py     # AUC-ROC, calibration, margin-aware accuracy
│   ├── analyze_ensemble_nt.py        # Run ensemble on all 27 NT books
│   ├── analyze_multiscale.py         # Multi-granularity analysis
│   ├── detect_redactions.py          # Automated redaction detection + robustness validation
│   └── detect_extended_redactions.py # Catholic + expanded marker set analysis
├── results/
│   ├── ensemble/                     # Full clause-level NT results (by_book/, gentile_extraction/, pattern_analysis/)
│   ├── redaction/                    # Original redaction detection + robustness report
│   ├── extended_redaction/           # Catholic + expanded marker set results
│   └── multiscale/                   # Cross-scale comparison (token_window/, clause/, verse/, block_2v/, block_3v/)
└── paper/
    └── paper.md                      # Paper source
```

## Quick Start

### Requirements

```bash
pip install torch numpy
```

### Training the Ensemble

The training corpus is included as `Corpus/training_corpus.tar.gz` (44 MB compressed, 291 MB extracted). It will be automatically extracted on first run.

- **Jewish tradition** (44 sources, 144 files): Philo, Josephus, pseudepigrapha, apostolic fathers, all parsed into PROIEL-compatible CoNLL-U
- **Gentile tradition** (9 sources, 100 files): Plutarch, Epictetus, Dio Chrysostom, Marcus Aurelius, Polybius, Aelius Aristides, and others

```bash
python analysis/train_ensemble.py
```

Trains 10 MorphTagTransformer models with seeds 42-51. Each model trains for 20 epochs with batch size 256, AdamW optimizer (lr=1e-3), and OneCycleLR schedule. Runs on CPU (~15 min per model on a modern machine). Pre-trained weights are included in `trained_models/`.

### Running NT Analysis

The 27 NT books in CoNLL-U format are included in `new-testament/`.

```bash
python analysis/analyze_ensemble_nt.py
```

Analyzes all 27 NT books at clause granularity. Outputs CSV, JSON, and human-readable reports.

### Redaction Detection

```bash
python analysis/detect_redactions.py
```

Strips editorial markers from Jewish-classified clauses, re-scores, and identifies tradition flips. Includes integrated random ablation validation (100 trials per flip) with per-flip p-values. Outputs:

- `robustness_report.txt`:Robust/rare/fragile classification with p-values for each flip
- `tradition_flips.txt`:Human-readable report of all flips sorted by robustness and delta
- `redaction_results.csv`:Complete data for all stripping operations
- `redaction_results.json`:Full data including per-model probabilities
- `category_ablation.csv`:Which editorial categories drive each flip
- `reconstructed_text.txt`:Stripped Greek text from flipped passages
- `book_summaries.txt`:Per-book flip rates and statistics
- `summary_statistics.txt`:Overall analysis summary

### Extended Redaction (Catholic + Expanded Markers)

```bash
python analysis/detect_extended_redactions.py
```

Runs the same redaction pipeline with two additional marker sets (Catholic institutional and expanded theological) in both J→G and G→J directions. Outputs to `results/extended_redaction/`.

## How It Works

### What the Model Sees

For the Greek text "ὁ θεὸς τοῦ αἰῶνος τούτου ἐτύφλωσεν τὰ νοήματα", the model receives:

| Token     | Morph Code   | Deprel |
| --------- | ------------ | ------ |
| ὁ         | `M--–––mn–s` | det    |
| θεὸς      | `N--–––mn–-` | nsubj  |
| τοῦ       | `M--–––ng–s` | det    |
| αἰῶνος    | `N--–––mg–-` | nmod   |
| τούτου    | `P--–––mg–-` | det    |
| ἐτύφλωσεν | `V3s-a-i–––` | root   |
| τὰ        | `M--–––an–s` | det    |
| νοήματα   | `N--–––an–-` | obj    |

It sees grammatical structure (case chains, verb morphology, syntactic roles) but never the words themselves. "θεός" and "ἄνθρωπος" look identical if they have the same morphological features.

### Redaction Detection Logic

1. Score a clause through the ensemble → P(Jewish) = 0.508
2. Identify editorial markers by lemma (e.g., χριστός, κύριος)
3. Strip matched tokens + their syntactic dependents (articles, modifiers)
4. Re-score the remaining tokens → P(Jewish) = 0.352
5. If P(J) crossed 0.5 threshold: **tradition flip detected**

## Citation

```bibtex
@article{turcotte2026morphosyntactic,
  title={Morphosyntactic Fingerprinting of Editorial Layers in the Greek New Testament},
  author={Turcotte, Jared Marshall},
  year={2026},
  doi={10.5281/zenodo.19078777},
  publisher={Zenodo},
  url={https://zenodo.org/records/19078777}
}
```

## License

MIT License
