"""
Multi-scale ensemble analysis of the entire NT.
Runs at 4 granularity levels:
  1. token_window  - sliding window of N tokens (stride N/2)
  2. clause        - split on Greek punctuation (current method)
  3. verse         - full CoNLL-U sentence units
  4. block         - 2-3 consecutive verses combined

Exports CSV + JSON for each level.

Usage:
    python analyze_multiscale.py
"""
import sys, json, csv, time
from collections import defaultdict
import torch, numpy as np, torch.nn.functional as F
from pathlib import Path

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
OUTPUT_DIR = ROOT / "results" / "multiscale"
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

TOKEN_WINDOW = 10
TOKEN_STRIDE = 5

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


# ── encoding ─────────────────────────────────────────────────────────────

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


class FakeSentence:
    def __init__(self, wt):
        self.word_tokens = wt
        self.sent_id = ""


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

    gp = extract_sentence_grammar_profile(FakeSentence(tokens))
    gp = np.array(gp, dtype=np.float32)

    return (
        torch.tensor(morph_seq).unsqueeze(0),
        torch.tensor(deprel_seq).unsqueeze(0),
        torch.tensor(morph_mask).unsqueeze(0),
        torch.tensor(gp).unsqueeze(0),
    )


# ── ensemble ─────────────────────────────────────────────────────────────

def load_ensemble():
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


def ensemble_predict(models, tokens):
    if len(tokens) == 0:
        return 0.5, 0.0, [0.5] * len(models)
    morph_seq, deprel_seq, morph_mask, gp = encode_tokens(tokens)
    all_pj = []
    with torch.no_grad():
        for model in models:
            logits = model(morph_seq, deprel_seq, morph_mask, gp)
            probs = F.softmax(logits, dim=-1)
            all_pj.append(probs[0, 0].item())
    all_pj = np.array(all_pj)
    return float(np.mean(all_pj)), float(np.std(all_pj)), all_pj.tolist()


# ── segmentation strategies ──────────────────────────────────────────────

def split_clauses(tokens, min_tokens=3):
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


def split_token_windows(tokens, window=TOKEN_WINDOW, stride=TOKEN_STRIDE):
    """Sliding window over tokens."""
    windows = []
    for start in range(0, len(tokens), stride):
        end = min(start + window, len(tokens))
        chunk = tokens[start:end]
        if len(chunk) >= 3:
            windows.append((start, end, chunk))
        if end >= len(tokens):
            break
    return windows


def split_verse_blocks(sentences, block_size):
    """Combine consecutive sentences into blocks."""
    blocks = []
    for i in range(0, len(sentences), block_size):
        group = sentences[i:i + block_size]
        combined_tokens = []
        chapters = []
        for s in group:
            combined_tokens.extend(s.word_tokens)
            ch = get_chapter(s)
            if ch not in chapters:
                chapters.append(ch)
        if combined_tokens:
            blocks.append({
                'tokens': combined_tokens[:MAX_MORPH_SEQ_LEN],
                'chapters': chapters,
                'verse_range': f"{get_chapter(group[0])}.{group[0].sent_id}-{get_chapter(group[-1])}.{group[-1].sent_id}",
                'sent_ids': [s.sent_id for s in group],
            })
    return blocks


# ── helpers ──────────────────────────────────────────────────────────────

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


def make_result(book_short, book_full, chapter, unit_idx, n_tokens, text,
                mean_pj, std_pj, per_model):
    confidence = "HIGH" if std_pj < 0.05 else ("LOW" if std_pj > 0.15 else "MED")
    return {
        'book': book_short,
        'book_full': book_full,
        'chapter': chapter,
        'unit_idx': unit_idx,
        'p_jewish': round(mean_pj, 4),
        'p_gentile': round(1 - mean_pj, 4),
        'uncertainty': round(std_pj, 4),
        'confidence': confidence,
        'label': 'Jewish' if mean_pj > 0.5 else 'Gentile',
        'per_model_pj': [round(p, 4) for p in per_model],
        'n_tokens': n_tokens,
        'text': text,
    }


# ── per-book analysis at all scales ──────────────────────────────────────

def analyze_book_multiscale(models, parser, file_prefix, short_name, full_name):
    conllu_path = find_conllu(file_prefix)
    if conllu_path is None:
        print(f"  WARNING: Could not find {file_prefix}")
        return {}, {}

    sentences = parser.parse_file(str(conllu_path))
    sentences.sort(key=get_chapter)

    # Filter out chapter 0
    sentences = [s for s in sentences if get_chapter(s) > 0]

    results = {
        'token_window': [],
        'clause': [],
        'verse': [],
        'block_2v': [],
        'block_3v': [],
    }

    # ── Level 1: Token windows (per verse, then pooled) ──
    for sent in sentences:
        ch = get_chapter(sent)
        windows = split_token_windows(sent.word_tokens)
        for wi, (start, end, chunk) in enumerate(windows):
            text = " ".join(t.form for t in chunk)
            mean_pj, std_pj, per_model = ensemble_predict(models, chunk)
            results['token_window'].append(
                make_result(short_name, full_name, ch, wi + 1,
                            len(chunk), text, mean_pj, std_pj, per_model)
            )

    # ── Level 2: Clauses (concatenate all tokens per chapter, then split) ──
    chapter_tokens = defaultdict(list)
    for sent in sentences:
        ch = get_chapter(sent)
        chapter_tokens[ch].extend(sent.word_tokens)
    for ch in sorted(chapter_tokens.keys()):
        clauses = split_clauses(chapter_tokens[ch])
        for ci, clause_tokens in enumerate(clauses):
            text = " ".join(t.form for t in clause_tokens)
            mean_pj, std_pj, per_model = ensemble_predict(models, clause_tokens)
            results['clause'].append(
                make_result(short_name, full_name, ch, ci + 1,
                            len(clause_tokens), text, mean_pj, std_pj, per_model)
            )

    # ── Level 3: Full verses/sentences ──
    for si, sent in enumerate(sentences):
        ch = get_chapter(sent)
        tokens = sent.word_tokens[:MAX_MORPH_SEQ_LEN]
        text = " ".join(t.form for t in tokens)
        mean_pj, std_pj, per_model = ensemble_predict(models, tokens)
        r = make_result(short_name, full_name, ch, si + 1,
                        len(tokens), text, mean_pj, std_pj, per_model)
        r['sent_id'] = sent.sent_id
        results['verse'].append(r)

    # ── Level 4a: 2-verse blocks ──
    blocks_2 = split_verse_blocks(sentences, 2)
    for bi, block in enumerate(blocks_2):
        ch = block['chapters'][0] if block['chapters'] else 0
        text = " ".join(t.form for t in block['tokens'])
        mean_pj, std_pj, per_model = ensemble_predict(models, block['tokens'])
        r = make_result(short_name, full_name, ch, bi + 1,
                        len(block['tokens']), text, mean_pj, std_pj, per_model)
        r['sent_ids'] = block['sent_ids']
        results['block_2v'].append(r)

    # ── Level 4b: 3-verse blocks ──
    blocks_3 = split_verse_blocks(sentences, 3)
    for bi, block in enumerate(blocks_3):
        ch = block['chapters'][0] if block['chapters'] else 0
        text = " ".join(t.form for t in block['tokens'])
        mean_pj, std_pj, per_model = ensemble_predict(models, block['tokens'])
        r = make_result(short_name, full_name, ch, bi + 1,
                        len(block['tokens']), text, mean_pj, std_pj, per_model)
        r['sent_ids'] = block['sent_ids']
        results['block_3v'].append(r)

    # Book-level summary per scale
    summaries = {}
    for scale, data in results.items():
        if data:
            pjs = [r['p_jewish'] for r in data]
            uncs = [r['uncertainty'] for r in data]
            n_j = sum(1 for p in pjs if p > 0.5)
            summaries[scale] = {
                'book': short_name,
                'book_full': full_name,
                'n_units': len(data),
                'mean_p_jewish': round(float(np.mean(pjs)), 4),
                'mean_uncertainty': round(float(np.mean(uncs)), 4),
                'n_jewish': n_j,
                'n_gentile': len(data) - n_j,
                'pct_jewish': round(100 * n_j / len(data), 1),
                'avg_tokens': round(np.mean([r['n_tokens'] for r in data]), 1),
            }

    return results, summaries


# ── export helpers ───────────────────────────────────────────────────────

CSV_FIELDS = [
    'book', 'book_full', 'chapter', 'unit_idx',
    'p_jewish', 'p_gentile', 'uncertainty', 'confidence',
    'label', 'n_tokens', 'text',
]


def export_scale(all_results, all_summaries, scale_name, scale_dir):
    """Export CSV, JSON, and summary for one scale."""
    scale_dir.mkdir(exist_ok=True)

    # CSV
    with open(scale_dir / f"{scale_name}.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    # JSON (full data including per_model_pj)
    with open(scale_dir / f"{scale_name}.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Book summaries
    with open(scale_dir / f"{scale_name}_book_summary.json", 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # Summary text
    with open(scale_dir / f"{scale_name}_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 90}\n")
        f.write(f"  SCALE: {scale_name}\n")
        f.write(f"  Total units: {len(all_results)}\n")
        if all_results:
            all_pjs = [r['p_jewish'] for r in all_results]
            all_uncs = [r['uncertainty'] for r in all_results]
            avg_tok = np.mean([r['n_tokens'] for r in all_results])
            f.write(f"  Avg tokens/unit: {avg_tok:.1f}\n")
            f.write(f"  Mean P(Jewish): {np.mean(all_pjs):.3f} +/- {np.mean(all_uncs):.3f}\n")
            n_j = sum(1 for p in all_pjs if p > 0.5)
            f.write(f"  Jewish: {n_j}/{len(all_pjs)} ({100*n_j/len(all_pjs):.1f}%)\n")
            f.write(f"  Gentile: {len(all_pjs)-n_j}/{len(all_pjs)} ({100*(len(all_pjs)-n_j)/len(all_pjs):.1f}%)\n")
        f.write(f"{'=' * 90}\n\n")

        f.write(f"  {'Book':<20s} {'Units':>6s} {'AvgTok':>7s} {'P(J)':>7s} {'Unc':>6s} "
                f"{'J/G':>10s} {'%J':>6s}  {'Signal':>8s}\n")
        f.write(f"  {'-'*20} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*10} {'-'*6}  {'-'*8}\n")

        for s in all_summaries:
            signal = "JEWISH" if s['mean_p_jewish'] > 0.55 else (
                "GENTILE" if s['mean_p_jewish'] < 0.45 else "MIXED")
            jg = f"{s['n_jewish']}/{s['n_gentile']}"
            f.write(f"  {s['book_full']:<20s} {s['n_units']:>6d} {s['avg_tokens']:>7.1f} "
                    f"{s['mean_p_jewish']:>7.3f} {s['mean_uncertainty']:>6.3f} "
                    f"{jg:>10s} {s['pct_jewish']:>5.1f}%  {signal:>8s}\n")


def write_cross_scale_comparison(scale_summaries, out_dir):
    """Compare book rankings across scales."""
    with open(out_dir / "cross_scale_comparison.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("  CROSS-SCALE COMPARISON: Book rankings at each granularity\n")
        f.write("=" * 100 + "\n\n")

        scales = ['token_window', 'clause', 'verse', 'block_2v', 'block_3v']
        scale_labels = {
            'token_window': f'Tokens ({TOKEN_WINDOW}w/{TOKEN_STRIDE}s)',
            'clause': 'Clause',
            'verse': 'Verse',
            'block_2v': '2-Verse Block',
            'block_3v': '3-Verse Block',
        }

        # Header
        f.write(f"  {'Book':<16s}")
        for s in scales:
            f.write(f"  {scale_labels[s]:>18s}")
        f.write("\n")
        f.write(f"  {'-'*16}")
        for _ in scales:
            f.write(f"  {'-'*18}")
        f.write("\n")

        # Get all books
        books = []
        if 'clause' in scale_summaries:
            books = [s['book_full'] for s in scale_summaries['clause']]

        for book_full in books:
            f.write(f"  {book_full:<16s}")
            for scale in scales:
                if scale in scale_summaries:
                    match = [s for s in scale_summaries[scale] if s['book_full'] == book_full]
                    if match:
                        s = match[0]
                        pj = s['mean_p_jewish']
                        label = "J" if pj > 0.55 else ("G" if pj < 0.45 else "M")
                        f.write(f"  {pj:.3f} {label} ±{s['mean_uncertainty']:.3f}")
                    else:
                        f.write(f"  {'N/A':>18s}")
                else:
                    f.write(f"  {'N/A':>18s}")
            f.write("\n")

        # Scale-level totals
        f.write(f"\n\n  {'TOTALS':<16s}")
        for scale in scales:
            if scale in scale_summaries:
                all_pj = [s['mean_p_jewish'] for s in scale_summaries[scale]]
                avg = np.mean(all_pj) if all_pj else 0
                f.write(f"  {'mean':>8s}={avg:.3f}     ")
            else:
                f.write(f"  {'N/A':>18s}")
        f.write("\n")

        # Stability analysis
        f.write(f"\n\n{'=' * 100}\n")
        f.write("  STABILITY: Books whose signal changes across scales\n")
        f.write("=" * 100 + "\n\n")

        for book_full in books:
            signals = {}
            for scale in scales:
                if scale in scale_summaries:
                    match = [s for s in scale_summaries[scale] if s['book_full'] == book_full]
                    if match:
                        pj = match[0]['mean_p_jewish']
                        signals[scale] = pj

            if signals:
                vals = list(signals.values())
                spread = max(vals) - min(vals)
                if spread > 0.05:
                    min_scale = min(signals, key=signals.get)
                    max_scale = max(signals, key=signals.get)
                    f.write(f"  {book_full:<20s} spread={spread:.3f}  "
                            f"most Jewish at {scale_labels[max_scale]} ({signals[max_scale]:.3f}), "
                            f"most Gentile at {scale_labels[min_scale]} ({signals[min_scale]:.3f})\n")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    models = load_ensemble()
    parser = ConlluParser()

    # Accumulate results per scale
    all_data = {
        'token_window': [],
        'clause': [],
        'verse': [],
        'block_2v': [],
        'block_3v': [],
    }
    all_summaries = {
        'token_window': [],
        'clause': [],
        'verse': [],
        'block_2v': [],
        'block_3v': [],
    }

    print(f"\nMulti-scale analysis: {len(BOOK_ORDER)} NT books x 5 scales x 10 models\n")

    for i, (prefix, short, full, category) in enumerate(BOOK_ORDER):
        t0 = time.time()
        results, summaries = analyze_book_multiscale(models, parser, prefix, short, full)
        elapsed = time.time() - t0

        for scale in all_data:
            if scale in results:
                all_data[scale].extend(results[scale])
            if scale in summaries:
                all_summaries[scale].append(summaries[scale])

        # Quick status from clause level
        if 'clause' in summaries:
            s = summaries['clause']
            signal = "JEWISH" if s['mean_p_jewish'] > 0.55 else (
                "GENTILE" if s['mean_p_jewish'] < 0.45 else "MIXED")
            counts = {sc: len(results.get(sc, [])) for sc in all_data}
            print(f"  {i+1:2d}/{len(BOOK_ORDER)}  {full:<20s} P(J)={s['mean_p_jewish']:.3f} {signal:<8s} "
                  f"tok={counts['token_window']:4d} cl={counts['clause']:3d} "
                  f"vs={counts['verse']:3d} b2={counts['block_2v']:3d} b3={counts['block_3v']:3d}  "
                  f"({elapsed:.1f}s)")

    # Export each scale
    print(f"\nExporting results...")
    for scale in all_data:
        scale_dir = OUTPUT_DIR / scale
        export_scale(all_data[scale], all_summaries[scale], scale, scale_dir)
        n = len(all_data[scale])
        avg_tok = np.mean([r['n_tokens'] for r in all_data[scale]]) if all_data[scale] else 0
        print(f"  {scale:<15s}: {n:6d} units, avg {avg_tok:.1f} tokens -> {scale_dir}/")

    # Cross-scale comparison
    write_cross_scale_comparison(all_summaries, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print(f"  All results saved to: {OUTPUT_DIR}/")
    print(f"  Subdirectories per scale:")
    for scale in all_data:
        print(f"    {scale}/  - CSV, JSON, summary")
    print(f"  cross_scale_comparison.txt - Book signals at each scale")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
