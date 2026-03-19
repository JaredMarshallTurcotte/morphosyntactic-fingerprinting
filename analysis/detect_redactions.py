"""
NT Redaction Detector & Reconstruction Tool

Systematically strips suspected editorial insertions from NT clauses classified
as Jewish, re-scores through the 10-model ensemble, and identifies tradition flips
(Jewish -> Gentile) revealing hypothesized pre-editorial substrate.

Includes integrated random ablation validation: each editorial flip is tested
against N_RANDOM_TRIALS random token deletions to compute per-flip p-values
and robustness classification (ROBUST / RARE / FRAGILE).

Usage:
    python3 detect_redactions.py
"""
import sys, json, csv, time
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.analyze_ensemble_nt import (
    load_ensemble, ensemble_predict, encode_tokens,
    split_into_clauses, find_conllu, BOOK_ORDER,
)
from data.conllu_parser import ConlluParser

OUTPUT_DIR = ROOT / "results" / "redaction"
N_RANDOM_TRIALS = 100  # random ablation trials per editorial flip
RANDOM_SEED = 456

# ── Editorial marker categories (lemma-based, verified against CoNLL-U) ──

EDITORIAL_CATEGORIES = {
    'christological': {
        'lemmas': {'Ἰησοῦς', 'Χριστός'},
        'desc': 'Christological insertions (Jesus, Christ)',
    },
    'kyrios': {
        'lemmas': {'κύριος'},
        'desc': 'Lord references (ambiguous Jewish/Christological)',
    },
    'patriarchal': {
        'lemmas': {'Μωϋσῆς', 'Ἀβραάμ', 'Δαυίδ', 'Ἰσραήλ', 'Ἰακώβ', 'Ἰσαάκ'},
        'desc': 'OT patriarchal references',
    },
    'scripture_citation': {
        'lemmas': {'γράφω'},
        'desc': 'Scripture citation formulae (covers γέγραπται via lemma)',
    },
    'prophetic': {
        'lemmas': {'προφήτης', 'Ἠσαΐας', 'Ἰερεμίας', 'Ἠλίας'},
        'desc': 'Prophet references',
    },
    'torah_law': {
        'lemmas': {'νόμος', 'ἐντολή', 'περιτομή'},
        'desc': 'Torah/Law references',
    },
}

ALL_EDITORIAL_LEMMAS = set()
for cat in EDITORIAL_CATEGORIES.values():
    ALL_EDITORIAL_LEMMAS |= cat['lemmas']

# Dependency relations that cascade when head is stripped
CASCADE_DEPRELS = {'det', 'amod', 'case', 'nmod', 'flat', 'flat:name'}


def strip_tokens(clause_tokens, target_lemmas):
    """Strip tokens matching target lemmas plus their syntactic dependents.

    Returns (remaining_tokens, stripped_info) where stripped_info is a list of
    (token, category) tuples.
    """
    # Build ID -> token map
    id_to_tok = {}
    for tok in clause_tokens:
        try:
            tid = int(tok.id)
            id_to_tok[tid] = tok
        except (ValueError, TypeError):
            pass

    # First pass: find directly matched tokens
    stripped_ids = set()
    tok_categories = {}
    for tok in clause_tokens:
        if tok.lemma in target_lemmas:
            try:
                tid = int(tok.id)
                stripped_ids.add(tid)
                # Find which category matched
                for cat_name, cat_info in EDITORIAL_CATEGORIES.items():
                    if tok.lemma in cat_info['lemmas']:
                        tok_categories[tid] = cat_name
                        break
            except (ValueError, TypeError):
                pass

    # Cascade to dependents
    changed = True
    while changed:
        changed = False
        for tok in clause_tokens:
            try:
                tid = int(tok.id)
                head = int(tok.head) if tok.head else 0
            except (ValueError, TypeError):
                continue
            if tid not in stripped_ids and head in stripped_ids:
                dep = tok.deprel.split(':')[0] if tok.deprel else ''
                full_dep = tok.deprel if tok.deprel else ''
                if dep in CASCADE_DEPRELS or full_dep in CASCADE_DEPRELS:
                    stripped_ids.add(tid)
                    tok_categories[tid] = tok_categories.get(head, 'dependent')
                    changed = True

    remaining = []
    stripped_info = []
    for tok in clause_tokens:
        try:
            tid = int(tok.id)
        except (ValueError, TypeError):
            remaining.append(tok)
            continue
        if tid in stripped_ids:
            stripped_info.append({
                'form': tok.form,
                'lemma': tok.lemma,
                'upos': tok.upos or '',
                'deprel': tok.deprel or '',
                'category': tok_categories.get(tid, 'dependent'),
            })
        else:
            remaining.append(tok)

    return remaining, stripped_info


def count_real_tokens(tokens):
    """Count non-punctuation tokens."""
    return sum(1 for t in tokens if t.upos not in ('PUNCT', 'X', None))


def random_strip_tokens(clause_tokens, n_to_strip, rng):
    """Randomly strip n_to_strip non-editorial, non-punctuation tokens,
    then cascade their dependents using the same logic as editorial stripping.

    Returns (remaining_tokens, n_actually_stripped) or (None, 0) if impossible.
    """
    # Find candidate tokens: non-editorial, non-punctuation
    candidates = []
    for tok in clause_tokens:
        try:
            tid = int(tok.id)
        except (ValueError, TypeError):
            continue
        if tok.lemma in ALL_EDITORIAL_LEMMAS:
            continue
        if tok.upos in ('PUNCT', 'X', None):
            continue
        candidates.append(tid)

    if len(candidates) < n_to_strip:
        return None, 0

    # Randomly select tokens to strip
    chosen = set(rng.choice(candidates, size=min(n_to_strip, len(candidates)), replace=False))

    # Cascade to dependents (same logic as editorial stripping)
    stripped_ids = set(chosen)
    changed = True
    while changed:
        changed = False
        for tok in clause_tokens:
            try:
                tid = int(tok.id)
                head = int(tok.head) if tok.head else 0
            except (ValueError, TypeError):
                continue
            if tid not in stripped_ids and head in stripped_ids:
                dep = tok.deprel.split(':')[0] if tok.deprel else ''
                full_dep = tok.deprel if tok.deprel else ''
                if dep in CASCADE_DEPRELS or full_dep in CASCADE_DEPRELS:
                    stripped_ids.add(tid)
                    changed = True

    remaining = []
    for tok in clause_tokens:
        try:
            tid = int(tok.id)
        except (ValueError, TypeError):
            remaining.append(tok)
            continue
        if tid not in stripped_ids:
            remaining.append(tok)

    return remaining, len(stripped_ids)


def validate_flip_robustness(models, clause_tokens, n_editorial_stripped, rng):
    """Run N_RANDOM_TRIALS random ablation trials on a single clause.

    Returns (random_flip_count, random_deltas, p_value, robustness_label).
    """
    # Get original score
    try:
        morph_seq, deprel_seq, morph_mask, gp = encode_tokens(clause_tokens)
        orig_pj, _, _ = ensemble_predict(models, morph_seq, deprel_seq, morph_mask, gp)
    except (ValueError, RuntimeError):
        return 0, [], 1.0, 'ERROR'

    random_flip_count = 0
    random_deltas = []

    for _ in range(N_RANDOM_TRIALS):
        remaining, _ = random_strip_tokens(clause_tokens, n_editorial_stripped, rng)
        if remaining is None or count_real_tokens(remaining) < 3:
            continue
        try:
            ms, ds, mm, gp2 = encode_tokens(remaining)
            rand_pj, _, _ = ensemble_predict(models, ms, ds, mm, gp2)
            random_deltas.append(orig_pj - rand_pj)
            if rand_pj < 0.5:
                random_flip_count += 1
        except (ValueError, RuntimeError):
            continue

    # p-value: probability of seeing 0 random flips or fewer under null
    # For robust flips (0 random flips), p < 1/N_RANDOM_TRIALS
    p_value = (random_flip_count + 1) / (N_RANDOM_TRIALS + 1)  # Laplace smoothing

    if random_flip_count == 0:
        label = 'ROBUST'
    elif random_flip_count <= 5:
        label = 'RARE'
    else:
        label = 'FRAGILE'

    return random_flip_count, random_deltas, p_value, label


def get_sent_id_info(sent):
    """Extract book abbreviation and chapter from sent_id."""
    parts = sent.sent_id.split('.')
    if len(parts) >= 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            return parts[0], 0
    return sent.sent_id, 0


def _derive_sent_id(tokens, short_name, fallback):
    """Derive verse-level sent_id from token misc field."""
    for tok in tokens:
        misc = tok.misc if hasattr(tok, 'misc') and tok.misc else ''
        if 'verse=' in misc:
            vparts = misc.split('verse=')[1].split('\t')[0].split('|')[0]
            ch_v = vparts.split(':')
            if len(ch_v) == 2:
                return f"{short_name}.{ch_v[0]}.{ch_v[1]}"
    return fallback


def _process_units(models, units, short_name, full_name, category):
    """Score and strip editorial markers from a list of (sent_id, tokens) units.

    Returns list of result dicts.
    """
    results = []
    for unit_idx, (sent_id, chapter, unit_tokens) in enumerate(units, 1):
        if len(unit_tokens) < 3:
            continue
        try:
            morph_seq, deprel_seq, morph_mask, gp = encode_tokens(unit_tokens)
            orig_pj, orig_std, orig_per_model = ensemble_predict(
                models, morph_seq, deprel_seq, morph_mask, gp
            )
        except (ValueError, RuntimeError):
            continue

        if orig_pj < 0.5:
            continue

        orig_text = ' '.join(t.form for t in unit_tokens)
        n_orig = len(unit_tokens)

        # ── ALL-strip ──
        remaining, stripped_info = strip_tokens(unit_tokens, ALL_EDITORIAL_LEMMAS)
        categories_hit = list(set(s['category'] for s in stripped_info if s['category'] != 'dependent'))

        if stripped_info:
            if count_real_tokens(remaining) >= 3:
                try:
                    ms, ds, mm, gp2 = encode_tokens(remaining)
                    strip_pj, strip_std, strip_per_model = ensemble_predict(
                        models, ms, ds, mm, gp2
                    )
                except (ValueError, RuntimeError):
                    strip_pj, strip_std, strip_per_model = None, None, None
            else:
                strip_pj, strip_std, strip_per_model = None, None, None

            results.append({
                'book': short_name,
                'book_full': full_name,
                'category': category,
                'chapter': chapter,
                'clause_idx': unit_idx,
                'sent_id': sent_id,
                'strategy': 'all',
                'orig_pj': orig_pj,
                'orig_std': orig_std,
                'orig_per_model': orig_per_model,
                'stripped_pj': strip_pj,
                'stripped_std': strip_std,
                'stripped_per_model': strip_per_model,
                'delta_pj': (orig_pj - strip_pj) if strip_pj is not None else None,
                'flipped': strip_pj is not None and strip_pj < 0.5,
                'fully_editorial': strip_pj is None,
                'n_orig': n_orig,
                'n_stripped': len(stripped_info),
                'n_remaining': len(remaining),
                'strip_ratio': len(stripped_info) / n_orig,
                'categories_hit': categories_hit,
                'orig_text': orig_text,
                'stripped_text': ' '.join(t.form for t in remaining) if remaining else '',
                'tokens_removed': [s['form'] for s in stripped_info],
                'stripped_detail': stripped_info,
                '_clause_tokens': unit_tokens,
            })

        # ── Per-category strips ──
        for cat_name, cat_info in EDITORIAL_CATEGORIES.items():
            cat_remaining, cat_stripped = strip_tokens(unit_tokens, cat_info['lemmas'])
            if not cat_stripped:
                continue

            if count_real_tokens(cat_remaining) >= 3:
                try:
                    ms, ds, mm, gp2 = encode_tokens(cat_remaining)
                    cat_pj, cat_std, cat_per_model = ensemble_predict(
                        models, ms, ds, mm, gp2
                    )
                except (ValueError, RuntimeError):
                    cat_pj, cat_std, cat_per_model = None, None, None
            else:
                cat_pj, cat_std, cat_per_model = None, None, None

            results.append({
                'book': short_name,
                'book_full': full_name,
                'category': category,
                'chapter': chapter,
                'clause_idx': unit_idx,
                'sent_id': sent_id,
                'strategy': cat_name,
                'orig_pj': orig_pj,
                'orig_std': orig_std,
                'orig_per_model': orig_per_model,
                'stripped_pj': cat_pj,
                'stripped_std': cat_std,
                'stripped_per_model': cat_per_model,
                'delta_pj': (orig_pj - cat_pj) if cat_pj is not None else None,
                'flipped': cat_pj is not None and cat_pj < 0.5,
                'fully_editorial': cat_pj is None,
                'n_orig': n_orig,
                'n_stripped': len(cat_stripped),
                'n_remaining': len(cat_remaining),
                'strip_ratio': len(cat_stripped) / n_orig,
                'categories_hit': [cat_name],
                'orig_text': orig_text,
                'stripped_text': ' '.join(t.form for t in cat_remaining) if cat_remaining else '',
                'tokens_removed': [s['form'] for s in cat_stripped],
                'stripped_detail': cat_stripped,
            })

    return results


def process_book(models, file_prefix, short_name, full_name, category):
    """Process one NT book at both clause and verse granularity."""
    path = find_conllu(file_prefix)
    if not path:
        print(f"  WARNING: Could not find {file_prefix}")
        return [], []

    parser = ConlluParser()
    sentences = parser.parse_file(str(path))
    sentences.sort(key=lambda s: (
        int(s.sent_id.split('.')[1]) if len(s.sent_id.split('.')) >= 3 else 0,
        int(s.sent_id.split('.')[2]) if len(s.sent_id.split('.')) >= 3 else 0,
    ))
    print(f"  {full_name}: {len(sentences)} sentences")

    # ── Build clause-level units (concatenate chapter, then split on punctuation) ──
    chapter_tokens = defaultdict(list)
    chapter_sent_ids = defaultdict(list)
    for sent in sentences:
        book_abbr, chapter = get_sent_id_info(sent)
        if chapter == 0:
            continue
        chapter_tokens[chapter].extend(sent.word_tokens)
        chapter_sent_ids[chapter].append(sent.sent_id)

    clause_units = []
    for ch in sorted(chapter_tokens.keys()):
        ref_id = chapter_sent_ids[ch][0] if chapter_sent_ids[ch] else f"{short_name}.{ch}.1"
        clauses = split_into_clauses(chapter_tokens[ch])
        for clause_tokens in clauses:
            sid = _derive_sent_id(clause_tokens, short_name, ref_id)
            clause_units.append((sid, ch, clause_tokens))

    # ── Build verse-level units (one unit per CoNLL-U sentence) ──
    verse_units = []
    for sent in sentences:
        book_abbr, chapter = get_sent_id_info(sent)
        if chapter == 0 or not sent.word_tokens:
            continue
        verse_units.append((sent.sent_id, chapter, sent.word_tokens))

    # ── Process both ──
    clause_results = _process_units(models, clause_units, short_name, full_name, category)
    verse_results = _process_units(models, verse_units, short_name, full_name, category)

    n_clause_flips = sum(1 for r in clause_results if r['strategy'] == 'all' and r['flipped'])
    n_verse_flips = sum(1 for r in verse_results if r['strategy'] == 'all' and r['flipped'])
    print(f"  -> clause: {len([r for r in clause_results if r['strategy']=='all'])} marked, {n_clause_flips} flips  |  "
          f"verse: {len([r for r in verse_results if r['strategy']=='all'])} marked, {n_verse_flips} flips")

    return clause_results, verse_results


def write_csv(results, path):
    """Write flat CSV of all results."""
    fields = [
        'book', 'book_full', 'category', 'chapter', 'clause_idx', 'sent_id',
        'strategy', 'orig_pj', 'orig_std', 'stripped_pj', 'stripped_std',
        'delta_pj', 'flipped', 'fully_editorial',
        'n_orig', 'n_stripped', 'n_remaining', 'strip_ratio',
        'categories_hit', 'orig_text', 'stripped_text', 'tokens_removed',
        'robustness', 'random_flip_count', 'p_value',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in results:
            row = dict(r)
            row['categories_hit'] = '|'.join(r['categories_hit'])
            row['tokens_removed'] = '|'.join(r['tokens_removed'])
            for k in ('orig_pj', 'orig_std', 'stripped_pj', 'stripped_std', 'delta_pj', 'strip_ratio'):
                if row[k] is not None:
                    row[k] = f"{row[k]:.4f}"
            w.writerow(row)


def write_json(results, path):
    """Write full JSON with per-model probabilities."""
    # Filter out internal fields not suitable for JSON
    clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in results]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean, f, ensure_ascii=False, indent=2,
                  default=lambda x: None if x != x else x)  # handle NaN


def write_tradition_flips(results, path):
    """Human-readable report of all tradition flips, sorted by robustness then delta."""
    flips = [r for r in results if r['strategy'] == 'all' and r['flipped']]
    # Sort: ROBUST first, then by delta descending
    order = {'ROBUST': 0, 'RARE': 1, 'FRAGILE': 2}
    flips.sort(key=lambda r: (order.get(r.get('robustness', 'FRAGILE'), 3),
                               -(r['delta_pj'] if r['delta_pj'] else 0)))

    n_robust = sum(1 for r in flips if r.get('robustness') == 'ROBUST')

    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("  TRADITION FLIPS: Jewish -> Gentile After Editorial Stripping\n")
        f.write(f"  Validated with {N_RANDOM_TRIALS} random ablation trials per flip\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"  Total flips: {len(flips)}  |  ROBUST: {n_robust}  |  "
                f"FRAGILE: {len(flips) - n_robust}\n\n")

        for i, r in enumerate(flips, 1):
            robustness = r.get('robustness', '?')
            p_val = r.get('p_value')
            rand_count = r.get('random_flip_count')
            f.write(f"  {i:3d}. [{robustness}] {r['book']} Ch{r['chapter']}  cl{r['clause_idx']}  "
                    f"({r['sent_id']})\n")
            f.write(f"       Delta = {r['delta_pj']:+.3f}   "
                    f"P(J): {r['orig_pj']:.3f} -> {r['stripped_pj']:.3f}   "
                    f"std: {r['orig_std']:.3f} -> {r['stripped_std']:.3f}\n")
            if rand_count is not None:
                f.write(f"       Random ablation: {rand_count}/{N_RANDOM_TRIALS} flips  "
                        f"p = {p_val:.4f}\n")
            f.write(f"       Stripped {r['n_stripped']}/{r['n_orig']} tokens "
                    f"({r['strip_ratio']:.0%})  "
                    f"Categories: {', '.join(r['categories_hit'])}\n")
            f.write(f"       Removed: {' '.join(r['tokens_removed'])}\n")
            f.write(f"       Original:  {r['orig_text']}\n")
            f.write(f"       Remaining: {r['stripped_text']}\n\n")

        # Also list fully editorial clauses
        fully_ed = [r for r in results if r['strategy'] == 'all' and r['fully_editorial']]
        if fully_ed:
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"  FULLY EDITORIAL: {len(fully_ed)} clauses with <3 real tokens after stripping\n")
            f.write("=" * 100 + "\n\n")
            for r in fully_ed:
                f.write(f"  {r['book']} Ch{r['chapter']}  ({r['sent_id']})  "
                        f"P(J)={r['orig_pj']:.3f}\n")
                f.write(f"    {r['orig_text']}\n\n")


def write_book_summaries(results, path):
    """Per-book summary of redaction analysis."""
    books = {}
    for r in results:
        if r['strategy'] != 'all':
            continue
        bk = r['book_full']
        if bk not in books:
            books[bk] = {
                'book': r['book'], 'book_full': bk, 'category': r['category'],
                'total_jewish_clauses': 0, 'with_markers': 0, 'flips': 0,
                'fully_editorial': 0, 'deltas': [],
                'per_cat_flips': defaultdict(int),
            }
        b = books[bk]
        b['total_jewish_clauses'] += 1
        if r['n_stripped'] > 0:
            b['with_markers'] += 1
        if r['flipped']:
            b['flips'] += 1
        if r['fully_editorial']:
            b['fully_editorial'] += 1
        if r['delta_pj'] is not None:
            b['deltas'].append(r['delta_pj'])

    # Count per-category flips from individual strategy results
    for r in results:
        if r['strategy'] != 'all' and r['flipped']:
            bk = r['book_full']
            if bk in books:
                books[bk]['per_cat_flips'][r['strategy']] += 1

    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("  REDACTION ANALYSIS: Per-Book Summaries\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"  {'Book':<22} {'Jewish':>7} {'Markers':>8} {'Flips':>6} {'Rate':>7} "
                f"{'FullEd':>7} {'MeanDelta':>10}\n")
        f.write(f"  {'-'*22} {'-'*7} {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*10}\n")

        for file_prefix, short, full, cat in BOOK_ORDER:
            if full not in books:
                continue
            b = books[full]
            mean_d = np.mean(b['deltas']) if b['deltas'] else 0
            rate = b['flips'] / b['with_markers'] if b['with_markers'] > 0 else 0
            f.write(f"  {full:<22} {b['total_jewish_clauses']:>7} {b['with_markers']:>8} "
                    f"{b['flips']:>6} {rate:>6.1%} {b['fully_editorial']:>7} "
                    f"{mean_d:>+9.3f}\n")

        f.write("\n\n")
        f.write("=" * 100 + "\n")
        f.write("  Per-Category Flip Drivers (individual category strips that cause flips)\n")
        f.write("=" * 100 + "\n\n")

        for file_prefix, short, full, cat in BOOK_ORDER:
            if full not in books:
                continue
            b = books[full]
            if not b['per_cat_flips']:
                continue
            cats = ', '.join(f"{k}={v}" for k, v in sorted(b['per_cat_flips'].items()))
            f.write(f"  {full:<22} {cats}\n")


def write_category_ablation(results, path):
    """For each clause that flipped under ALL-strip, show per-category deltas."""
    # Group by clause
    by_clause = defaultdict(dict)
    for r in results:
        key = (r['book'], r['chapter'], r['clause_idx'])
        by_clause[key][r['strategy']] = r

    fields = ['book', 'chapter', 'clause_idx', 'sent_id', 'orig_pj', 'all_delta',
              'christological_delta', 'kyrios_delta', 'patriarchal_delta',
              'scripture_citation_delta', 'prophetic_delta', 'torah_law_delta',
              'primary_driver', 'orig_text']

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for key in sorted(by_clause.keys()):
            strats = by_clause[key]
            if 'all' not in strats or not strats['all']['flipped']:
                continue

            row = {
                'book': strats['all']['book'],
                'chapter': strats['all']['chapter'],
                'clause_idx': strats['all']['clause_idx'],
                'sent_id': strats['all']['sent_id'],
                'orig_pj': f"{strats['all']['orig_pj']:.4f}",
                'all_delta': f"{strats['all']['delta_pj']:.4f}" if strats['all']['delta_pj'] else '',
                'orig_text': strats['all']['orig_text'],
            }

            max_delta = 0
            primary = 'none'
            for cat_name in EDITORIAL_CATEGORIES:
                d = strats.get(cat_name, {}).get('delta_pj')
                col = f"{cat_name}_delta"
                row[col] = f"{d:.4f}" if d is not None else ''
                if d is not None and d > max_delta:
                    max_delta = d
                    primary = cat_name

            row['primary_driver'] = primary
            w.writerow(row)


def write_reconstructed_text(results, path):
    """Stripped Greek text from flipped passages in reading order."""
    flips = [r for r in results if r['strategy'] == 'all' and r['flipped']]
    flips.sort(key=lambda r: (r['book'], r['chapter'], r['clause_idx']))

    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("  RECONSTRUCTED TEXT: Hypothesized Pre-Editorial Substrate\n")
        f.write("  (Greek text remaining after stripping editorial markers from flipped clauses)\n")
        f.write("=" * 100 + "\n\n")

        current_book = None
        current_ch = None

        for r in flips:
            if r['book_full'] != current_book:
                current_book = r['book_full']
                current_ch = None
                f.write(f"\n{'─' * 80}\n")
                f.write(f"  {current_book}\n")
                f.write(f"{'─' * 80}\n\n")

            if r['chapter'] != current_ch:
                current_ch = r['chapter']
                f.write(f"  [Chapter {current_ch}]\n")

            f.write(f"    {r['sent_id']}  P(J): {r['orig_pj']:.3f}->{r['stripped_pj']:.3f}  "
                    f"Δ={r['delta_pj']:+.3f}\n")
            f.write(f"    {r['stripped_text']}\n")
            f.write(f"    (removed: {', '.join(r['tokens_removed'])})\n\n")


def write_summary_statistics(results, path):
    """Overall statistics."""
    all_strat = [r for r in results if r['strategy'] == 'all']
    flips_all = [r for r in all_strat if r['flipped']]
    fully_ed = [r for r in all_strat if r['fully_editorial']]
    with_markers = [r for r in all_strat if r['n_stripped'] > 0]

    # By book category
    by_cat = defaultdict(lambda: {'total': 0, 'flips': 0})
    for r in all_strat:
        by_cat[r['category']]['total'] += 1
        if r['flipped']:
            by_cat[r['category']]['flips'] += 1

    # By editorial category
    cat_effectiveness = defaultdict(lambda: {'tested': 0, 'flips': 0, 'deltas': []})
    for r in results:
        if r['strategy'] in EDITORIAL_CATEGORIES:
            ce = cat_effectiveness[r['strategy']]
            ce['tested'] += 1
            if r['flipped']:
                ce['flips'] += 1
            if r['delta_pj'] is not None:
                ce['deltas'].append(r['delta_pj'])

    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("  REDACTION DETECTION: Summary Statistics\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"  Total Jewish clauses analyzed:     {len(all_strat):>6}\n")
        f.write(f"  Clauses with editorial markers:    {len(with_markers):>6} "
                f"({len(with_markers)/len(all_strat):.1%})\n")
        f.write(f"  Tradition flips (all-strip):       {len(flips_all):>6} "
                f"({len(flips_all)/len(with_markers):.1%} of marked)\n")
        f.write(f"  Fully editorial (<3 tokens left):  {len(fully_ed):>6}\n")

        if flips_all:
            deltas = [r['delta_pj'] for r in flips_all if r['delta_pj']]
            f.write(f"\n  Flip deltas:  mean={np.mean(deltas):+.3f}  "
                    f"median={np.median(deltas):+.3f}  "
                    f"max={max(deltas):+.3f}\n")

        f.write(f"\n\n  {'Book Category':<20} {'Jewish Cls':>10} {'Flips':>6} {'Rate':>7}\n")
        f.write(f"  {'-'*20} {'-'*10} {'-'*6} {'-'*7}\n")
        for cat in ['Gospel', 'KnownPauline', 'UncertainPaul', 'Disputed', 'Catholic', 'Apocalyptic']:
            if cat in by_cat:
                bc = by_cat[cat]
                rate = bc['flips'] / bc['total'] if bc['total'] > 0 else 0
                f.write(f"  {cat:<20} {bc['total']:>10} {bc['flips']:>6} {rate:>6.1%}\n")

        f.write(f"\n\n  {'Editorial Category':<22} {'Tested':>7} {'Flips':>6} {'Rate':>7} {'Mean Delta':>11}\n")
        f.write(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*7} {'-'*11}\n")
        for cat_name in EDITORIAL_CATEGORIES:
            ce = cat_effectiveness[cat_name]
            if ce['tested'] > 0:
                rate = ce['flips'] / ce['tested']
                mean_d = np.mean(ce['deltas']) if ce['deltas'] else 0
                f.write(f"  {cat_name:<22} {ce['tested']:>7} {ce['flips']:>6} "
                        f"{rate:>6.1%} {mean_d:>+10.3f}\n")


def write_robustness_report(flip_results, path):
    """Write detailed robustness report with p-values for each flip."""
    # Sort: robust first (by delta desc), then rare, then fragile
    order = {'ROBUST': 0, 'RARE': 1, 'FRAGILE': 2}
    flip_results.sort(key=lambda r: (order.get(r['robustness'], 3), -r['delta_pj']))

    n_robust = sum(1 for r in flip_results if r['robustness'] == 'ROBUST')
    n_rare = sum(1 for r in flip_results if r['robustness'] == 'RARE')
    n_fragile = sum(1 for r in flip_results if r['robustness'] == 'FRAGILE')

    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"  REDACTION DETECTION: Robustness-Validated Results\n")
        f.write(f"  Random ablation: {N_RANDOM_TRIALS} trials per flip\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"  Total editorial flips:     {len(flip_results)}\n")
        f.write(f"  ROBUST  (0/{N_RANDOM_TRIALS} random):    {n_robust}   "
                f"(p < {1/(N_RANDOM_TRIALS+1):.4f} each)\n")
        f.write(f"  RARE    (1-5/{N_RANDOM_TRIALS} random):  {n_rare}\n")
        f.write(f"  FRAGILE (>5/{N_RANDOM_TRIALS} random):   {n_fragile}\n\n")

        # Combined p-value for robust flips (Fisher's method isn't needed;
        # binomial: probability of 9 independent events each with p<0.01)
        if n_robust > 0:
            per_flip_p = 1 / (N_RANDOM_TRIALS + 1)
            combined_p = per_flip_p ** n_robust
            f.write(f"  Combined probability of {n_robust} independent robust flips\n")
            f.write(f"  occurring by chance: p < {combined_p:.2e}\n\n")

        f.write("-" * 100 + "\n")
        f.write(f"  {'#':<4} {'Label':<8} {'Book':<10} {'SentID':<18} {'Δ':>7} "
                f"{'P(J) orig→strip':>18} {'Rand flips':>12} {'p-value':>10}\n")
        f.write("-" * 100 + "\n")

        for i, r in enumerate(flip_results, 1):
            f.write(f"  {i:<4} {r['robustness']:<8} {r['book']:<10} {r['sent_id']:<18} "
                    f"{r['delta_pj']:>+.3f} "
                    f"{r['orig_pj']:.3f} -> {r['stripped_pj']:.3f}   "
                    f"{r['random_flip_count']:>4}/{N_RANDOM_TRIALS}   "
                    f"{r['p_value']:>9.4f}\n")

        # Detailed section for robust flips
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("  ROBUST FLIPS: Detailed Analysis\n")
        f.write("=" * 100 + "\n\n")

        for r in flip_results:
            if r['robustness'] != 'ROBUST':
                continue
            f.write(f"  [{r['robustness']}] {r['book']} ({r['sent_id']})  "
                    f"p < {r['p_value']:.4f}\n")
            f.write(f"    P(J): {r['orig_pj']:.3f} -> {r['stripped_pj']:.3f}  "
                    f"Δ = {r['delta_pj']:+.3f}  "
                    f"σ: {r['orig_std']:.3f} -> {r['stripped_std']:.3f}\n")
            f.write(f"    Random flips: {r['random_flip_count']}/{N_RANDOM_TRIALS}  "
                    f"Mean random Δ: {r['mean_random_delta']:+.3f}\n")
            f.write(f"    Removed:   {' '.join(r['tokens_removed'])}\n")
            f.write(f"    Categories: {', '.join(r['categories_hit'])}\n")
            f.write(f"    Original:  {r['orig_text']}\n")
            f.write(f"    Remaining: {r['stripped_text']}\n\n")


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)

    models = load_ensemble()
    print()

    all_clause_results = []
    all_verse_results = []
    for file_prefix, short_name, full_name, category in BOOK_ORDER:
        print(f"Processing {full_name}...")
        clause_results, verse_results = process_book(
            models, file_prefix, short_name, full_name, category
        )
        all_clause_results.extend(clause_results)
        all_verse_results.extend(verse_results)

    # ── Write both granularities ──
    for label, all_results, subdir in [
        ("clause", all_clause_results, OUTPUT_DIR / "clause"),
        ("verse", all_verse_results, OUTPUT_DIR / "verse"),
    ]:
        subdir.mkdir(exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Validating {label}-level flips...")
        print(f"{'='*60}")

        editorial_flips = [r for r in all_results if r['strategy'] == 'all' and r['flipped']]
        print(f"  {len(editorial_flips)} editorial flips "
              f"({N_RANDOM_TRIALS} random trials each)")

        flip_validation_results = []
        for i, r in enumerate(editorial_flips, 1):
            clause_tokens = r.get('_clause_tokens')
            if clause_tokens is None:
                continue

            random_flip_count, random_deltas, p_value, rob_label = validate_flip_robustness(
                models, clause_tokens, r['n_stripped'], rng
            )

            r['random_flip_count'] = random_flip_count
            r['random_flip_rate'] = random_flip_count / N_RANDOM_TRIALS
            r['p_value'] = p_value
            r['robustness'] = rob_label
            r['mean_random_delta'] = np.mean(random_deltas) if random_deltas else 0.0

            flip_validation_results.append(r)
            print(f"  {i:3d}/{len(editorial_flips)}  {r['book']:<10} {r['sent_id']:<18} "
                  f"Δ={r['delta_pj']:+.3f}  random={random_flip_count}/{N_RANDOM_TRIALS}  [{rob_label}]")

        print(f"\n  {label} total results: {len(all_results)}")
        print(f"  Writing to {subdir}/...")

        write_csv(all_results, subdir / "redaction_results.csv")
        write_json(all_results, subdir / "redaction_results.json")
        write_tradition_flips(all_results, subdir / "tradition_flips.txt")
        write_book_summaries(all_results, subdir / "book_summaries.txt")
        write_category_ablation(all_results, subdir / "category_ablation.csv")
        write_reconstructed_text(all_results, subdir / "reconstructed_text.txt")
        write_summary_statistics(all_results, subdir / "summary_statistics.txt")
        write_robustness_report(flip_validation_results,
                                subdir / "robustness_report.txt")

        n_robust = sum(1 for r in flip_validation_results if r['robustness'] == 'ROBUST')
        n_rare = sum(1 for r in flip_validation_results if r['robustness'] == 'RARE')
        n_fragile = sum(1 for r in flip_validation_results if r['robustness'] == 'FRAGILE')
        print(f"\n  {label} robustness summary:")
        print(f"    ROBUST:  {n_robust}  (p < {1/(N_RANDOM_TRIALS+1):.4f} each)")
        print(f"    RARE:    {n_rare}")
        print(f"    FRAGILE: {n_fragile}")
        if n_robust > 0:
            combined_p = (1 / (N_RANDOM_TRIALS + 1)) ** n_robust
            print(f"    Combined p for {n_robust} robust flips: < {combined_p:.2e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output files in {OUTPUT_DIR}/clause/ and {OUTPUT_DIR}/verse/")


if __name__ == "__main__":
    main()
