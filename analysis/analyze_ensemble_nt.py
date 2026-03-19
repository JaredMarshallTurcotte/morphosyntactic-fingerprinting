"""
Full New Testament grammar tradition analysis using deep ensemble (10 models).
Averages softmax outputs across models, computes uncertainty via disagreement.

Usage:
    python analyze_ensemble_nt.py
"""
import sys, json, csv, time
import torch, numpy as np, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.conllu_parser import ConlluParser
from data.morphology_normalizer import MorphologyNormalizer
from data.feature_extractor import (
    extract_sentence_grammar_profile, SENTENCE_GRAMMAR_PROFILE_DIM,
)
from data.constants import (
    PROIEL_POSITION_VOCABS, MORPH_CODE_DIM, MAX_MORPH_SEQ_LEN,
    DEPREL_TO_IDX, DEPREL_PAD_IDX,
)
from models.morph_transformer import MorphTagTransformer

ENSEMBLE_DIR = ROOT / "trained_models"
OUTPUT_DIR = ROOT / "results" / "ensemble"

ENSEMBLE_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

BOOK_ORDER = [
    ("61-Mt",  "Matt",     "Matthew",          "Gospel"),
    ("62-Mk",  "Mark",     "Mark",             "Gospel"),
    ("63-Lk",  "Luke",     "Luke",             "Gospel"),
    ("64-Jn",  "John",     "John",             "Gospel"),
    ("65-Ac",  "Acts",     "Acts",             "Disputed"),
    ("66-Ro",  "Rom",      "Romans",           "KnownPauline"),
    ("67-1Co", "1Cor",     "1 Corinthians",    "KnownPauline"),
    ("68-2Co", "2Cor",     "2 Corinthians",    "KnownPauline"),
    ("69-Ga",  "Gal",      "Galatians",        "KnownPauline"),
    ("70-Eph", "Eph",      "Ephesians",        "UncertainPaul"),
    ("71-Php", "Phil",     "Philippians",      "KnownPauline"),
    ("72-Col", "Col",      "Colossians",       "UncertainPaul"),
    ("73-1Th", "1Thess",   "1 Thessalonians",  "KnownPauline"),
    ("74-2Th", "2Thess",   "2 Thessalonians",  "UncertainPaul"),
    ("75-1Ti", "1Tim",     "1 Timothy",        "Disputed"),
    ("76-2Ti", "2Tim",     "2 Timothy",        "Disputed"),
    ("77-Tit", "Titus",    "Titus",            "Disputed"),
    ("78-Phm", "Phlm",     "Philemon",         "KnownPauline"),
    ("79-Heb", "Heb",      "Hebrews",          "Disputed"),
    ("80-Jas", "Jas",      "James",            "Catholic"),
    ("81-1Pe", "1Pet",     "1 Peter",          "Catholic"),
    ("82-2Pe", "2Pet",     "2 Peter",          "Catholic"),
    ("83-1Jn", "1John",    "1 John",           "Catholic"),
    ("84-2Jn", "2John",    "2 John",           "Catholic"),
    ("85-3Jn", "3John",    "3 John",           "Catholic"),
    ("86-Jud", "Jude",     "Jude",             "Catholic"),
    ("87-Re",  "Rev",      "Revelation",       "Apocalyptic"),
]


def encode_morph_code(code):
    vec = []
    for pos_idx, vocab in enumerate(PROIEL_POSITION_VOCABS):
        one_hot = [0.0] * len(vocab)
        if pos_idx < len(code):
            char = code[pos_idx]
            if char in vocab:
                one_hot[vocab.index(char)] = 1.0
        vec.extend(one_hot)
    return vec


def encode_tokens(tokens):
    n = min(len(tokens), MAX_MORPH_SEQ_LEN)
    morph_seq = np.zeros((MAX_MORPH_SEQ_LEN, MORPH_CODE_DIM), dtype=np.float32)
    deprel_seq = np.full(MAX_MORPH_SEQ_LEN, DEPREL_PAD_IDX, dtype=np.int64)
    morph_mask = np.zeros(MAX_MORPH_SEQ_LEN, dtype=bool)
    for i in range(n):
        tok = tokens[i]
        m = MorphologyNormalizer.conllu_to_features(tok)
        pc = MorphologyNormalizer.features_to_proiel(m)
        morph_seq[i] = encode_morph_code(pc)
        dr = tok.deprel.split(':')[0] if tok.deprel else 'other'
        deprel_seq[i] = DEPREL_TO_IDX.get(dr, DEPREL_TO_IDX['other'])
        morph_mask[i] = True

    class FakeSentence:
        def __init__(self, wt):
            self.word_tokens = wt
            self.sent_id = ""
    gp = extract_sentence_grammar_profile(FakeSentence(tokens))
    gp = np.array(gp, dtype=np.float32)

    return (
        torch.tensor(morph_seq).unsqueeze(0),
        torch.tensor(deprel_seq).unsqueeze(0),
        torch.tensor(morph_mask).unsqueeze(0),
        torch.tensor(gp).unsqueeze(0),
    )


def split_into_clauses(tokens, min_tokens=3):
    clauses = []
    current = []
    for tok in tokens:
        current.append(tok)
        form = tok.form.rstrip('\u2e03\u2e05')
        if form and form[-1] in '.\u00b7;':
            if len(current) >= min_tokens:
                clauses.append(current)
                current = []
    if current and len(current) >= min_tokens:
        clauses.append(current)
    elif current and clauses:
        clauses[-1].extend(current)
    return clauses


def find_conllu(file_prefix):
    nt_dir = ROOT / "new-testament"
    for subdir in nt_dir.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.name.startswith(file_prefix) and f.suffix == ".conllu":
                    return f
    return None


def get_chapter(sent):
    parts = sent.sent_id.split('.')
    return int(parts[1]) if len(parts) >= 3 else 0


def get_verse(sent):
    parts = sent.sent_id.split('.')
    return int(parts[2]) if len(parts) >= 3 else 0


def load_ensemble():
    """Load all 10 ensemble models."""
    models = []
    print("Loading ensemble models...")
    for seed in ENSEMBLE_SEEDS:
        path = ENSEMBLE_DIR / f"grammar_ensemble_seed{seed}.pt"
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        model = MorphTagTransformer(
            morph_dim=MORPH_CODE_DIM,
            deprel_vocab_size=len(DEPREL_TO_IDX),
            sentence_profile_dim=SENTENCE_GRAMMAR_PROFILE_DIM,
            num_classes=2,
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"  Seed {seed}: val_acc={ckpt['val_acc']:.4f}")
        models.append(model)
    return models


def ensemble_predict(models, morph_seq, deprel_seq, morph_mask, gp):
    """Run all models, return mean P(Jewish), std, and per-model probs."""
    all_pj = []
    with torch.no_grad():
        for model in models:
            logits = model(morph_seq, deprel_seq, morph_mask, gp)
            probs = F.softmax(logits, dim=-1)
            all_pj.append(probs[0, 0].item())
    all_pj = np.array(all_pj)
    return float(np.mean(all_pj)), float(np.std(all_pj)), all_pj.tolist()


def analyze_book(models, parser, file_prefix, short_name, full_name):
    conllu_path = find_conllu(file_prefix)
    if conllu_path is None:
        print(f"  WARNING: Could not find {file_prefix}")
        return []

    sentences = parser.parse_file(str(conllu_path))
    sentences.sort(key=lambda s: (get_chapter(s), get_verse(s)))

    # Group all tokens by chapter, then clause-split per chapter
    chapter_tokens = defaultdict(list)
    for sent in sentences:
        ch_num = get_chapter(sent)
        if ch_num == 0:
            continue
        chapter_tokens[ch_num].extend(sent.word_tokens)

    results = []
    for ch_num in sorted(chapter_tokens.keys()):
        all_tokens = chapter_tokens[ch_num]
        clauses = split_into_clauses(all_tokens)
        for ci, clause_tokens in enumerate(clauses):
            text = " ".join(t.form for t in clause_tokens)
            n_tok = len(clause_tokens)

            morph_seq, deprel_seq, morph_mask, gp = encode_tokens(clause_tokens)
            mean_pj, std_pj, per_model = ensemble_predict(
                models, morph_seq, deprel_seq, morph_mask, gp
            )

            # Confidence: high when all models agree, low when they disagree
            confidence = "HIGH" if std_pj < 0.05 else ("LOW" if std_pj > 0.15 else "MED")

            results.append({
                'book': short_name,
                'book_full': full_name,
                'chapter': ch_num,
                'clause_idx': ci + 1,
                'p_jewish': round(mean_pj, 4),
                'p_gentile': round(1 - mean_pj, 4),
                'uncertainty': round(std_pj, 4),
                'confidence': confidence,
                'label': 'Jewish' if mean_pj > 0.5 else 'Gentile',
                'per_model_pj': [round(p, 4) for p in per_model],
                'n_tokens': n_tok,
                'text': text,
            })

    return results


def write_book_report(results, book_short, book_full, category, out_dir):
    if not results:
        return

    with open(out_dir / f"{book_short}.txt", 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 90}\n")
        f.write(f"  {book_full} - Deep Ensemble Grammar Tradition Analysis (10 models)\n")
        f.write(f"  Category: {category}\n")
        f.write(f"{'=' * 90}\n\n")

        chapters = sorted(set(r['chapter'] for r in results))
        for ch in chapters:
            ch_results = [r for r in results if r['chapter'] == ch]
            f.write(f"  Chapter {ch}\n")
            f.write(f"  {'-' * 80}\n")
            for r in ch_results:
                pj = r['p_jewish']
                label = "J" if pj > 0.5 else "G"
                bar_len = int(20 * pj)
                bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
                conf = r['confidence']
                display = r['text'][:75] + "..." if len(r['text']) > 75 else r['text']
                f.write(f"  {r['clause_idx']:3d}. P(J)={pj:.3f} +/-{r['uncertainty']:.3f} "
                        f"[{bar}] {label} {conf:<4s} {display}\n")
            f.write("\n")

        f.write(f"\n{'=' * 90}\n")
        f.write(f"  CHAPTER SUMMARY\n")
        f.write(f"{'=' * 90}\n")
        f.write(f"  {'Ch':>4s}  {'Mean P(J)':>9s}  {'Unc':>5s}  {'J/Total':>8s}  {'Signal':>8s}\n")
        f.write(f"  {'----':>4s}  {'---------':>9s}  {'-----':>5s}  {'-------':>8s}  {'------':>8s}\n")

        for ch in chapters:
            ch_results = [r for r in results if r['chapter'] == ch]
            pjs = [r['p_jewish'] for r in ch_results]
            uncs = [r['uncertainty'] for r in ch_results]
            mean_pj = np.mean(pjs)
            mean_unc = np.mean(uncs)
            n_j = sum(1 for p in pjs if p > 0.5)
            signal = "JEWISH" if mean_pj > 0.55 else ("GENTILE" if mean_pj < 0.45 else "MIXED")
            f.write(f"  {ch:4d}  {mean_pj:9.3f}  {mean_unc:5.3f}  {n_j:3d}/{len(pjs):<3d}  {signal:>8s}\n")

        all_pjs = [r['p_jewish'] for r in results]
        all_uncs = [r['uncertainty'] for r in results]
        f.write(f"\n  Overall: Mean P(J)={np.mean(all_pjs):.3f} +/-{np.mean(all_uncs):.3f}, "
                f"Jewish={sum(1 for p in all_pjs if p > 0.5)}/{len(all_pjs)}, "
                f"Gentile={sum(1 for p in all_pjs if p <= 0.5)}/{len(all_pjs)}\n")


def write_gentile_extraction(all_results, out_dir):
    """Extract and analyze all Gentile-classified clauses."""
    gent_dir = out_dir / "gentile_extraction"
    gent_dir.mkdir(exist_ok=True)

    gentile = [r for r in all_results if r['label'] == 'Gentile']
    strong_gentile = [r for r in gentile if r['p_jewish'] < 0.3]
    high_conf_gentile = [r for r in gentile if r['confidence'] != 'LOW']

    # All gentile clauses
    with open(gent_dir / "all_gentile_clauses.txt", 'w', encoding='utf-8') as f:
        f.write(f"ALL GENTILE-CLASSIFIED CLAUSES ({len(gentile)} total)\n")
        f.write(f"High confidence (all models agree): {len(high_conf_gentile)}\n")
        f.write("=" * 90 + "\n\n")
        cur_book = None
        for r in gentile:
            if r['book_full'] != cur_book:
                cur_book = r['book_full']
                f.write(f"\n  --- {cur_book} ---\n")
            conf = r['confidence']
            f.write(f"  Ch{r['chapter']:2d} cl{r['clause_idx']:2d}  "
                    f"P(J)={r['p_jewish']:.3f} +/-{r['uncertainty']:.3f} {conf:<4s}  "
                    f"{r['text'][:90]}\n")

    # Strong gentile (P(J) < 0.3) ranked by confidence
    strong_sorted = sorted(strong_gentile, key=lambda r: (r['uncertainty'], r['p_jewish']))
    with open(gent_dir / "strong_gentile_ranked.txt", 'w', encoding='utf-8') as f:
        f.write(f"STRONG GENTILE CLAUSES - P(J) < 0.3 ({len(strong_gentile)} total)\n")
        f.write(f"Sorted by uncertainty (most confident first)\n")
        f.write("=" * 90 + "\n\n")
        for r in strong_sorted:
            f.write(f"  {r['book']:<6s} Ch{r['chapter']:2d} cl{r['clause_idx']:2d}  "
                    f"P(J)={r['p_jewish']:.3f} +/-{r['uncertainty']:.3f} {r['confidence']:<4s}  "
                    f"{r['text'][:85]}\n")

    # Gentile clusters (3+ consecutive Gentile clauses)
    with open(gent_dir / "gentile_clusters.txt", 'w', encoding='utf-8') as f:
        f.write("GENTILE CLUSTERS (3+ consecutive Gentile clauses within a book)\n")
        f.write("=" * 90 + "\n\n")

        for prefix, short, full, cat in BOOK_ORDER:
            book_results = [r for r in all_results if r['book'] == short]
            if not book_results:
                continue

            runs = []
            run_start = 0
            for i in range(1, len(book_results)):
                if book_results[i]['label'] != book_results[run_start]['label']:
                    if book_results[run_start]['label'] == 'Gentile' and i - run_start >= 3:
                        runs.append((run_start, i))
                    run_start = i
            if book_results[run_start]['label'] == 'Gentile' and len(book_results) - run_start >= 3:
                runs.append((run_start, len(book_results)))

            if runs:
                f.write(f"\n  --- {full} ---\n")
                for start, end in runs:
                    cluster = book_results[start:end]
                    pjs = [r['p_jewish'] for r in cluster]
                    uncs = [r['uncertainty'] for r in cluster]
                    chapters = sorted(set(r['chapter'] for r in cluster))
                    ch_str = f"Ch {chapters[0]}" if len(chapters) == 1 else f"Ch {chapters[0]}-{chapters[-1]}"
                    f.write(f"\n  GENTILE CLUSTER: {len(cluster)} clauses ({ch_str})\n")
                    f.write(f"  Mean P(J)={np.mean(pjs):.3f} +/-{np.mean(uncs):.3f}\n")
                    for r in cluster:
                        display = r['text'][:80] + "..." if len(r['text']) > 80 else r['text']
                        f.write(f"    Ch{r['chapter']:2d} cl{r['clause_idx']:2d} "
                                f"P(J)={r['p_jewish']:.3f} +/-{r['uncertainty']:.3f}  {display}\n")

    # Chapter heatmap
    with open(gent_dir / "chapter_heatmap.txt", 'w', encoding='utf-8') as f:
        f.write("CHAPTER-LEVEL HEATMAP (Mean P(Jewish) | Uncertainty)\n")
        f.write("=" * 90 + "\n\n")
        for prefix, short, full, cat in BOOK_ORDER:
            book_results = [r for r in all_results if r['book'] == short]
            if not book_results:
                continue
            f.write(f"  {full}\n")
            chapters = sorted(set(r['chapter'] for r in book_results))
            for ch in chapters:
                ch_data = [r for r in book_results if r['chapter'] == ch]
                pjs = [r['p_jewish'] for r in ch_data]
                uncs = [r['uncertainty'] for r in ch_data]
                mean_pj = np.mean(pjs)
                mean_unc = np.mean(uncs)
                bar_len = int(30 * mean_pj)
                bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
                signal = "JEWISH" if mean_pj > 0.55 else ("GENTILE" if mean_pj < 0.45 else "MIXED")
                f.write(f"    Ch{ch:3d} [{bar}] {mean_pj:.3f} +/-{mean_unc:.3f} {signal}\n")
            f.write("\n")

    # JSON exports
    with open(gent_dir / "gentile_clauses.json", 'w', encoding='utf-8') as f:
        json.dump(gentile, f, ensure_ascii=False, indent=2)
    with open(gent_dir / "strong_gentile_clauses.json", 'w', encoding='utf-8') as f:
        json.dump(strong_sorted, f, ensure_ascii=False, indent=2)


def write_pattern_analysis(all_results, out_dir):
    """Look for organic patterns in the Gentile clauses."""
    pat_dir = out_dir / "pattern_analysis"
    pat_dir.mkdir(exist_ok=True)

    gentile = [r for r in all_results if r['label'] == 'Gentile']
    jewish = [r for r in all_results if r['label'] == 'Jewish']

    with open(pat_dir / "pattern_report.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("  PATTERN ANALYSIS - WHAT DISTINGUISHES GENTILE-CLASSIFIED NT PASSAGES?\n")
        f.write("  Deep Ensemble (10 models), averaged softmax with uncertainty\n")
        f.write("=" * 90 + "\n\n")

        # 1. Distribution by book category
        f.write("1. GENTILE CLAUSE DISTRIBUTION BY NT CATEGORY\n")
        f.write("-" * 60 + "\n")
        categories = ["Gospel", "KnownPauline", "UncertainPaul", "Disputed", "Catholic", "Apocalyptic"]
        cat_labels = {
            "Gospel": "Gospels", "KnownPauline": "Known Pauline",
            "UncertainPaul": "Uncertain Pauline", "Disputed": "Disputed",
            "Catholic": "Catholic Epistles", "Apocalyptic": "Apocalyptic",
        }
        for cat in categories:
            cat_books = [b[1] for b in BOOK_ORDER if b[3] == cat]
            total_cat = len([r for r in all_results if r['book'] in cat_books])
            gent_cat = len([r for r in gentile if r['book'] in cat_books])
            if total_cat > 0:
                pct = 100 * gent_cat / total_cat
                f.write(f"  {cat_labels[cat]:<22s}: {gent_cat:4d}/{total_cat:<4d} Gentile ({pct:.1f}%)\n")

        # 2. Books with highest Gentile %
        f.write(f"\n\n2. BOOKS RANKED BY GENTILE PERCENTAGE\n")
        f.write("-" * 60 + "\n")
        book_stats = {}
        for r in all_results:
            if r['book'] not in book_stats:
                book_stats[r['book']] = {'total': 0, 'gentile': 0, 'pjs': [], 'uncs': [],
                                          'full': r['book_full']}
            book_stats[r['book']]['total'] += 1
            book_stats[r['book']]['pjs'].append(r['p_jewish'])
            book_stats[r['book']]['uncs'].append(r['uncertainty'])
            if r['label'] == 'Gentile':
                book_stats[r['book']]['gentile'] += 1

        ranked = sorted(book_stats.items(),
                        key=lambda x: x[1]['gentile'] / max(x[1]['total'], 1), reverse=True)
        for book, stats in ranked:
            pct = 100 * stats['gentile'] / stats['total']
            mean_unc = np.mean(stats['uncs'])
            bar_len = int(30 * (1 - np.mean(stats['pjs'])))
            bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
            f.write(f"  {stats['full']:<20s} [{bar}] {pct:5.1f}% Gentile  "
                    f"(unc={mean_unc:.3f})  {stats['gentile']}/{stats['total']}\n")

        # 3. High-confidence Gentile passages (uncertainty < 0.05, P(J) < 0.3)
        hc_gentile = sorted(
            [r for r in all_results if r['p_jewish'] < 0.3 and r['uncertainty'] < 0.05],
            key=lambda r: r['p_jewish']
        )
        f.write(f"\n\n3. HIGHEST-CONFIDENCE GENTILE CLAUSES ({len(hc_gentile)} total)\n")
        f.write(f"   (P(J) < 0.3 AND all 10 models agree, uncertainty < 0.05)\n")
        f.write("-" * 60 + "\n")
        for r in hc_gentile[:50]:
            display = r['text'][:75] + "..." if len(r['text']) > 75 else r['text']
            f.write(f"  {r['book']:<6s} Ch{r['chapter']:2d}  "
                    f"P(J)={r['p_jewish']:.3f} +/-{r['uncertainty']:.3f}  {display}\n")
        if len(hc_gentile) > 50:
            f.write(f"  ... and {len(hc_gentile) - 50} more\n")

        # 4. High-confidence Jewish passages for contrast
        hc_jewish = sorted(
            [r for r in all_results if r['p_jewish'] > 0.7 and r['uncertainty'] < 0.05],
            key=lambda r: -r['p_jewish']
        )
        f.write(f"\n\n4. HIGHEST-CONFIDENCE JEWISH CLAUSES ({len(hc_jewish)} total)\n")
        f.write(f"   (P(J) > 0.7 AND all 10 models agree)\n")
        f.write("-" * 60 + "\n")
        for r in hc_jewish[:30]:
            display = r['text'][:75] + "..." if len(r['text']) > 75 else r['text']
            f.write(f"  {r['book']:<6s} Ch{r['chapter']:2d}  "
                    f"P(J)={r['p_jewish']:.3f} +/-{r['uncertainty']:.3f}  {display}\n")
        if len(hc_jewish) > 30:
            f.write(f"  ... and {len(hc_jewish) - 30} more\n")

        # 5. Mixed/uncertain passages (models disagree)
        uncertain = sorted(
            [r for r in all_results if r['uncertainty'] > 0.15],
            key=lambda r: -r['uncertainty']
        )
        f.write(f"\n\n5. MOST UNCERTAIN CLAUSES - MODELS DISAGREE ({len(uncertain)} total)\n")
        f.write(f"   (uncertainty > 0.15 = high model disagreement)\n")
        f.write("-" * 60 + "\n")
        for r in uncertain[:30]:
            per_model_str = " ".join(f"{p:.2f}" for p in r['per_model_pj'])
            display = r['text'][:65] + "..." if len(r['text']) > 65 else r['text']
            f.write(f"  {r['book']:<6s} Ch{r['chapter']:2d}  "
                    f"P(J)={r['p_jewish']:.3f} +/-{r['uncertainty']:.3f}  "
                    f"[{per_model_str}]  {display}\n")

        # 6. Transition zones - where Jewish passages border Gentile passages
        f.write(f"\n\n6. TRADITION TRANSITIONS (Jewish->Gentile or Gentile->Jewish boundaries)\n")
        f.write("-" * 60 + "\n")
        for prefix, short, full, cat in BOOK_ORDER:
            book_results = [r for r in all_results if r['book'] == short]
            if len(book_results) < 2:
                continue

            transitions = []
            for i in range(1, len(book_results)):
                prev_label = book_results[i-1]['label']
                curr_label = book_results[i]['label']
                if prev_label != curr_label:
                    # Only report high-confidence transitions
                    if (book_results[i-1]['confidence'] != 'LOW' and
                        book_results[i]['confidence'] != 'LOW'):
                        transitions.append((i-1, i))

            if transitions:
                f.write(f"\n  --- {full} ({len(transitions)} transitions) ---\n")
                for pi, ci in transitions[:10]:
                    prev = book_results[pi]
                    curr = book_results[ci]
                    direction = f"{prev['label']}->{curr['label']}"
                    prev_text = prev['text'][:50] + "..." if len(prev['text']) > 50 else prev['text']
                    curr_text = curr['text'][:50] + "..." if len(curr['text']) > 50 else curr['text']
                    f.write(f"    {direction} at Ch{curr['chapter']}\n")
                    f.write(f"      {prev['label'][0]}: P(J)={prev['p_jewish']:.3f}  {prev_text}\n")
                    f.write(f"      {curr['label'][0]}: P(J)={curr['p_jewish']:.3f}  {curr_text}\n")

    print(f"  Pattern analysis saved to {pat_dir}/")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    books_dir = OUTPUT_DIR / "by_book"
    books_dir.mkdir(exist_ok=True)

    models = load_ensemble()

    parser = ConlluParser()
    all_results = []
    book_summaries = []

    print(f"\nAnalyzing {len(BOOK_ORDER)} NT books with 10-model ensemble...\n")

    for i, (prefix, short, full, category) in enumerate(BOOK_ORDER):
        t0 = time.time()
        results = analyze_book(models, parser, prefix, short, full)
        elapsed = time.time() - t0

        if results:
            all_results.extend(results)
            pjs = [r['p_jewish'] for r in results]
            uncs = [r['uncertainty'] for r in results]
            mean_pj = np.mean(pjs)
            mean_unc = np.mean(uncs)
            n_j = sum(1 for p in pjs if p > 0.5)

            book_summaries.append({
                'book': short,
                'book_full': full,
                'category': category,
                'n_clauses': len(results),
                'mean_p_jewish': round(float(mean_pj), 4),
                'mean_uncertainty': round(float(mean_unc), 4),
                'n_jewish': n_j,
                'n_gentile': len(results) - n_j,
                'pct_jewish': round(100 * n_j / len(results), 1),
            })

            signal = "JEWISH" if mean_pj > 0.55 else ("GENTILE" if mean_pj < 0.45 else "MIXED")
            bar_len = int(20 * mean_pj)
            bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
            print(f"  {i+1:2d}/{len(BOOK_ORDER)}  {full:<20s} [{bar}] P(J)={mean_pj:.3f} +/-{mean_unc:.3f}  "
                  f"{n_j:3d}/{len(results):<3d} J  {signal:<8s}  ({elapsed:.1f}s)")

            write_book_report(results, short, full, category, books_dir)

    # Write all clauses
    csv_path = OUTPUT_DIR / "all_clauses.csv"
    csv_fields = [
        'book', 'book_full', 'chapter', 'clause_idx',
        'p_jewish', 'p_gentile', 'uncertainty', 'confidence',
        'label', 'n_tokens', 'text'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    with open(OUTPUT_DIR / "all_clauses.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "book_summaries.json", 'w', encoding='utf-8') as f:
        json.dump(book_summaries, f, ensure_ascii=False, indent=2)

    # Master summary
    report_path = OUTPUT_DIR / "NT_SUMMARY.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("  NEW TESTAMENT - DEEP ENSEMBLE GRAMMAR TRADITION ANALYSIS\n")
        f.write("  10x MorphTagTransformer (seeds 42-51), averaged softmax + uncertainty\n")
        f.write("  Corpus: PrunedCorpus (no Septuagint, genre-balanced Gentile)\n")
        f.write("=" * 90 + "\n\n")

        categories = ["Gospel", "KnownPauline", "UncertainPaul", "Disputed",
                       "Catholic", "Apocalyptic"]
        cat_labels = {
            "Gospel": "GOSPELS", "KnownPauline": "KNOWN PAULINE",
            "UncertainPaul": "UNCERTAIN PAULINE", "Disputed": "DISPUTED ATTRIBUTION",
            "Catholic": "CATHOLIC EPISTLES", "Apocalyptic": "APOCALYPTIC",
        }

        for cat in categories:
            cat_books = [b for b in book_summaries if b['category'] == cat]
            if not cat_books:
                continue
            f.write(f"\n  {cat_labels[cat]}\n")
            f.write(f"  {'-' * 80}\n")
            f.write(f"  {'Book':<20s} {'Clauses':>8s} {'Mean P(J)':>10s} {'Unc':>6s} "
                    f"{'J/G':>10s} {'%Jewish':>8s}  {'Signal':>8s}\n")

            for b in cat_books:
                signal = "JEWISH" if b['mean_p_jewish'] > 0.55 else (
                    "GENTILE" if b['mean_p_jewish'] < 0.45 else "MIXED")
                jg = f"{b['n_jewish']}/{b['n_gentile']}"
                f.write(f"  {b['book_full']:<20s} {b['n_clauses']:>8d} {b['mean_p_jewish']:>10.3f} "
                        f"{b['mean_uncertainty']:>6.3f} {jg:>10s} {b['pct_jewish']:>7.1f}%  "
                        f"{signal:>8s}\n")

            cat_pjs = [b['mean_p_jewish'] for b in cat_books]
            cat_uncs = [b['mean_uncertainty'] for b in cat_books]
            f.write(f"  {'AVERAGE':<20s} {'':>8s} {np.mean(cat_pjs):>10.3f} {np.mean(cat_uncs):>6.3f}\n")

        all_pjs = [r['p_jewish'] for r in all_results]
        all_uncs = [r['uncertainty'] for r in all_results]
        f.write(f"\n\n{'=' * 90}\n")
        f.write(f"  OVERALL NT STATISTICS\n")
        f.write(f"{'=' * 90}\n")
        f.write(f"  Total clauses analyzed: {len(all_results)}\n")
        f.write(f"  Mean P(Jewish): {np.mean(all_pjs):.3f} +/- {np.mean(all_uncs):.3f}\n")
        f.write(f"  Jewish clauses: {sum(1 for p in all_pjs if p > 0.5)}/{len(all_pjs)} "
                f"({100*sum(1 for p in all_pjs if p > 0.5)/len(all_pjs):.1f}%)\n")
        f.write(f"  Gentile clauses: {sum(1 for p in all_pjs if p <= 0.5)}/{len(all_pjs)} "
                f"({100*sum(1 for p in all_pjs if p <= 0.5)/len(all_pjs):.1f}%)\n")

        # High confidence counts
        hc_j = sum(1 for r in all_results if r['p_jewish'] > 0.7 and r['uncertainty'] < 0.05)
        hc_g = sum(1 for r in all_results if r['p_jewish'] < 0.3 and r['uncertainty'] < 0.05)
        f.write(f"  High-confidence Jewish (P>0.7, unc<0.05): {hc_j}\n")
        f.write(f"  High-confidence Gentile (P<0.3, unc<0.05): {hc_g}\n")

        f.write(f"\n\n  BOOKS RANKED BY P(JEWISH)\n")
        f.write(f"  {'-' * 80}\n")
        ranked = sorted(book_summaries, key=lambda b: b['mean_p_jewish'], reverse=True)
        for rank, b in enumerate(ranked, 1):
            bar_len = int(30 * b['mean_p_jewish'])
            bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
            f.write(f"  {rank:2d}. {b['book_full']:<20s} [{bar}] {b['mean_p_jewish']:.3f} "
                    f"+/-{b['mean_uncertainty']:.3f}  ({b['category']})\n")

    # Gentile extraction
    write_gentile_extraction(all_results, OUTPUT_DIR)

    # Pattern analysis
    write_pattern_analysis(all_results, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print(f"  Results saved to: {OUTPUT_DIR}/")
    print(f"    NT_SUMMARY.txt          - Master summary")
    print(f"    book_summaries.json      - Book-level stats")
    print(f"    all_clauses.csv          - Every clause with scores")
    print(f"    all_clauses.json         - Full data (JSON)")
    print(f"    by_book/                 - Per-book reports")
    print(f"    gentile_extraction/      - Gentile clause analysis")
    print(f"    pattern_analysis/        - Pattern discovery")
    print(f"{'=' * 60}")

    # Console ranking
    print(f"\n  BOOKS RANKED BY P(JEWISH):")
    print(f"  {'-' * 70}")
    for rank, b in enumerate(ranked, 1):
        bar_len = int(20 * b['mean_p_jewish'])
        bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
        signal = "JEWISH" if b['mean_p_jewish'] > 0.55 else (
            "GENTILE" if b['mean_p_jewish'] < 0.45 else "MIXED")
        print(f"  {rank:2d}. {b['book_full']:<20s} [{bar}] {b['mean_p_jewish']:.3f} "
              f"+/-{b['mean_uncertainty']:.3f}  {signal:<8s}  ({b['category']})")


if __name__ == "__main__":
    main()
