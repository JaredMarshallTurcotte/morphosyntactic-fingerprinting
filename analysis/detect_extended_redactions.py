"""
Extended Redaction Detection: Catholic and Expanded Marker Sets

Runs the same strip-and-rescore methodology as detect_redactions.py but with
two additional, independent marker sets:
  1. Catholic: ecclesiological, sacramental, creedal, hamartiology,
     pneumatological, eschatological
  2. Expanded: divine_reference, holiness, soteriology, faith

Includes random ablation validation (100 trials per flip) and tests both
J→G and G→J directions for Catholic markers.

Usage:
    python3 detect_extended_redactions.py
"""
import sys, json, csv, time
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.analyze_ensemble_nt import (
    load_ensemble, ensemble_predict, encode_tokens,
    split_into_clauses, find_conllu, BOOK_ORDER, ConlluParser
)
from analysis.detect_redactions import (
    strip_tokens, count_real_tokens, random_strip_tokens,
    N_RANDOM_TRIALS
)

OUTPUT_DIR = ROOT / "results" / "extended_redaction"
RANDOM_SEED = 789

# ── Catholic editorial marker categories ──

CATHOLIC_CATEGORIES = {
    'ecclesiological': {
        'lemmas': {'ἐκκλησία', 'ἐπίσκοπος', 'πρεσβύτερος', 'διάκονος', 'ἀπόστολος'},
        'desc': 'Church structure (church, bishop, elder, deacon, apostle)',
    },
    'sacramental': {
        'lemmas': {'βαπτίζω', 'βάπτισμα'},
        'desc': 'Baptismal terminology',
    },
    'creedal': {
        'lemmas': {'ἀνάστασις', 'σταυρός', 'σταυρόω'},
        'desc': 'Creedal (resurrection, cross, crucify)',
    },
    'hamartiology': {
        'lemmas': {'ἁμαρτία', 'μετάνοια', 'μετανοέω'},
        'desc': 'Sin/repentance terminology',
    },
    'pneumatological': {
        'lemmas': {'πνεῦμα'},
        'desc': 'Spirit references',
    },
    'eschatological': {
        'lemmas': {'παρουσία', 'βασιλεία'},
        'desc': 'Parousia/Kingdom',
    },
}

ALL_CATHOLIC_LEMMAS = set()
for cat in CATHOLIC_CATEGORIES.values():
    ALL_CATHOLIC_LEMMAS |= cat['lemmas']

# ── Expanded editorial marker categories ──

EXPANDED_CATEGORIES = {
    'divine_reference': {
        'lemmas': {'θεός', 'πατήρ'},
        'desc': 'God/Father references',
    },
    'holiness': {
        'lemmas': {'ἅγιος', 'δόξα'},
        'desc': 'Holy/glory terminology',
    },
    'soteriology': {
        'lemmas': {'σωτήρ', 'σωτηρία', 'εὐαγγέλιον'},
        'desc': 'Salvation/gospel terminology',
    },
    'faith': {
        'lemmas': {'πίστις'},
        'desc': 'Faith terminology',
    },
}

ALL_EXPANDED_LEMMAS = set()
for cat in EXPANDED_CATEGORIES.values():
    ALL_EXPANDED_LEMMAS |= cat['lemmas']


def collect_and_analyze(models, parser, marker_lemmas, marker_categories,
                        direction, rng, label):
    """Run full redaction analysis with given markers.

    direction: 'j_to_g' or 'g_to_j'
    Returns (all_results, flip_results_with_robustness)
    """
    all_results = []

    for file_prefix, short_name, full_name, category in BOOK_ORDER:
        path = find_conllu(file_prefix)
        if not path:
            continue
        sentences = parser.parse_file(str(path))
        sentences.sort(key=lambda s: (
            int(s.sent_id.split('.')[1]) if len(s.sent_id.split('.')) >= 3 else 0,
            int(s.sent_id.split('.')[2]) if len(s.sent_id.split('.')) >= 3 else 0,
        ))
        chapter_tokens = defaultdict(list)
        for sent in sentences:
            parts = sent.sent_id.split('.')
            ch = int(parts[1]) if len(parts) >= 3 else 0
            if ch == 0:
                continue
            chapter_tokens[ch].extend(sent.word_tokens)

        for ch in sorted(chapter_tokens.keys()):
            clauses = split_into_clauses(chapter_tokens[ch])
            for cl_idx, cl_tokens in enumerate(clauses):
                has_marker = any(t.lemma in marker_lemmas for t in cl_tokens)
                if not has_marker:
                    continue

                try:
                    morph_seq, deprel_seq, morph_mask, gp = encode_tokens(cl_tokens)
                    orig_pj, orig_std, _ = ensemble_predict(
                        models, morph_seq, deprel_seq, morph_mask, gp
                    )
                except (ValueError, RuntimeError):
                    continue

                if direction == 'j_to_g' and orig_pj < 0.5:
                    continue
                if direction == 'g_to_j' and orig_pj >= 0.5:
                    continue

                remaining, stripped_info = strip_tokens(cl_tokens, marker_lemmas)
                if not stripped_info:
                    continue

                cats_hit = []
                for cat_name, cat_info in marker_categories.items():
                    if any(t.lemma in cat_info['lemmas'] for t in cl_tokens):
                        cats_hit.append(cat_name)

                if count_real_tokens(remaining) < 3:
                    all_results.append({
                        'book': short_name, 'book_full': full_name,
                        'category': category, 'chapter': ch,
                        'orig_pj': orig_pj, 'orig_std': orig_std,
                        'fully_editorial': True, 'flipped': False,
                        'categories_hit': cats_hit,
                        'orig_text': ' '.join(t.form for t in cl_tokens),
                        '_clause_tokens': cl_tokens,
                    })
                    continue

                try:
                    ms, ds, mm, gp2 = encode_tokens(remaining)
                    strip_pj, strip_std, _ = ensemble_predict(models, ms, ds, mm, gp2)
                except (ValueError, RuntimeError):
                    continue

                delta = orig_pj - strip_pj
                if direction == 'j_to_g':
                    flipped = strip_pj < 0.5
                else:
                    flipped = strip_pj >= 0.5

                all_results.append({
                    'book': short_name, 'book_full': full_name,
                    'category': category, 'chapter': ch,
                    'orig_pj': orig_pj, 'orig_std': orig_std,
                    'strip_pj': strip_pj, 'strip_std': strip_std,
                    'delta': delta, 'flipped': flipped,
                    'fully_editorial': False,
                    'n_stripped': len(stripped_info),
                    'categories_hit': cats_hit,
                    'orig_text': ' '.join(t.form for t in cl_tokens),
                    'stripped_text': ' '.join(t.form for t in remaining),
                    'tokens_removed': [s['form'] for s in stripped_info],
                    '_clause_tokens': cl_tokens,
                })

    # Run ablation on flips
    flips = [r for r in all_results if r.get('flipped') and not r['fully_editorial']]
    print(f"\n  {label}: {len(flips)} flips found. Running ablation...")

    for i, flip in enumerate(flips):
        cl_tokens = flip['_clause_tokens']
        n_strip = flip['n_stripped']

        try:
            morph_seq, deprel_seq, morph_mask, gp = encode_tokens(cl_tokens)
            orig_pj, _, _ = ensemble_predict(models, morph_seq, deprel_seq, morph_mask, gp)
        except (ValueError, RuntimeError):
            flip['robustness'] = 'ERROR'
            continue

        random_flip_count = 0
        for _ in range(N_RANDOM_TRIALS):
            rem, _ = random_strip_tokens(cl_tokens, n_strip, rng)
            if rem is None or count_real_tokens(rem) < 3:
                continue
            try:
                ms, ds, mm, gp2 = encode_tokens(rem)
                rand_pj, _, _ = ensemble_predict(models, ms, ds, mm, gp2)
            except (ValueError, RuntimeError):
                continue
            if direction == 'j_to_g' and rand_pj < 0.5:
                random_flip_count += 1
            elif direction == 'g_to_j' and rand_pj >= 0.5:
                random_flip_count += 1

        p_value = (random_flip_count + 1) / (N_RANDOM_TRIALS + 1)
        if random_flip_count == 0:
            rob_label = 'ROBUST'
        elif random_flip_count <= 5:
            rob_label = 'RARE'
        else:
            rob_label = 'FRAGILE'

        flip['random_flip_count'] = random_flip_count
        flip['p_value'] = p_value
        flip['robustness'] = rob_label

        print(f"    {i+1:3d}/{len(flips)}  {flip['book']:8s} Ch{flip['chapter']:>2d}  "
              f"Δ={flip['delta']:+.3f}  random={random_flip_count:3d}/100  [{rob_label}]")

    return all_results, flips


def write_results(all_results, flips, output_dir, label):
    """Write results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scorable = [r for r in all_results if not r['fully_editorial']]
    n_robust = sum(1 for f in flips if f.get('robustness') == 'ROBUST')
    n_rare = sum(1 for f in flips if f.get('robustness') == 'RARE')
    n_fragile = sum(1 for f in flips if f.get('robustness') == 'FRAGILE')

    # Summary
    with open(output_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n  {label}: Summary\n{'='*80}\n\n")
        f.write(f"  Total clauses analyzed:  {len(all_results)}\n")
        f.write(f"  Scorable:               {len(scorable)}\n")
        f.write(f"  Fully editorial:        {len(all_results) - len(scorable)}\n")
        f.write(f"  Tradition flips:        {len(flips)}\n")
        if scorable:
            f.write(f"  Flip rate:              {len(flips)/len(scorable)*100:.1f}%\n")
        f.write(f"\n  Robustness:\n")
        f.write(f"    ROBUST:  {n_robust}\n")
        f.write(f"    RARE:    {n_rare}\n")
        f.write(f"    FRAGILE: {n_fragile}\n")
        if n_robust > 0:
            combined_p = (1 / (N_RANDOM_TRIALS + 1)) ** n_robust
            f.write(f"    Combined p for {n_robust} robust: < {combined_p:.2e}\n")

        f.write(f"\n  Robust flips:\n")
        for flip in sorted(flips, key=lambda x: -abs(x['delta'])):
            if flip.get('robustness') != 'ROBUST':
                continue
            f.write(f"    {flip['book']:8s} Ch{flip['chapter']:>2d}  "
                    f"Δ={flip['delta']:+.3f}  "
                    f"P(J): {flip['orig_pj']:.3f}→{flip['strip_pj']:.3f}  "
                    f"cats={','.join(flip['categories_hit'])}\n")
            f.write(f"      {flip['orig_text'][:100]}\n")

    # CSV
    fields = ['book', 'book_full', 'category', 'chapter', 'orig_pj', 'orig_std',
              'strip_pj', 'strip_std', 'delta', 'flipped', 'fully_editorial',
              'n_stripped', 'categories_hit', 'orig_text', 'stripped_text',
              'tokens_removed', 'robustness', 'random_flip_count', 'p_value']
    with open(output_dir / 'results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in all_results:
            row = dict(r)
            row['categories_hit'] = '|'.join(r.get('categories_hit', []))
            row['tokens_removed'] = '|'.join(r.get('tokens_removed', []))
            for k in ('orig_pj', 'orig_std', 'strip_pj', 'strip_std', 'delta'):
                if k in row and row[k] is not None:
                    row[k] = f"{row[k]:.4f}"
            w.writerow(row)

    # JSON (no internal fields)
    clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in all_results]
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(clean, f, ensure_ascii=False, indent=2,
                  default=lambda x: None if x != x else x)


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)

    models = load_ensemble()
    parser = ConlluParser()

    # Catholic J→G
    print("\n" + "=" * 80)
    print("  CATHOLIC J→G")
    print("=" * 80)
    cath_jg_all, cath_jg_flips = collect_and_analyze(
        models, parser, ALL_CATHOLIC_LEMMAS, CATHOLIC_CATEGORIES,
        'j_to_g', rng, 'Catholic J→G'
    )
    write_results(cath_jg_all, cath_jg_flips, OUTPUT_DIR / 'catholic_jg', 'Catholic J→G')

    # Catholic G→J
    print("\n" + "=" * 80)
    print("  CATHOLIC G→J")
    print("=" * 80)
    cath_gj_all, cath_gj_flips = collect_and_analyze(
        models, parser, ALL_CATHOLIC_LEMMAS, CATHOLIC_CATEGORIES,
        'g_to_j', rng, 'Catholic G→J'
    )
    write_results(cath_gj_all, cath_gj_flips, OUTPUT_DIR / 'catholic_gj', 'Catholic G→J')

    # Expanded J→G
    print("\n" + "=" * 80)
    print("  EXPANDED J→G")
    print("=" * 80)
    exp_all, exp_flips = collect_and_analyze(
        models, parser, ALL_EXPANDED_LEMMAS, EXPANDED_CATEGORIES,
        'j_to_g', rng, 'Expanded J→G'
    )
    write_results(exp_all, exp_flips, OUTPUT_DIR / 'expanded_jg', 'Expanded J→G')

    # Cross-analysis summary
    with open(OUTPUT_DIR / 'cross_analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  CROSS-ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        for lbl, flips in [
            ("Catholic J→G", cath_jg_flips),
            ("Catholic G→J", cath_gj_flips),
            ("Expanded J→G", exp_flips),
        ]:
            n_r = sum(1 for r in flips if r.get('robustness') == 'ROBUST')
            n_ra = sum(1 for r in flips if r.get('robustness') == 'RARE')
            n_f = sum(1 for r in flips if r.get('robustness') == 'FRAGILE')
            f.write(f"  {lbl}:\n")
            f.write(f"    Total flips: {len(flips)}\n")
            f.write(f"    ROBUST:  {n_r}\n")
            f.write(f"    RARE:    {n_ra}\n")
            f.write(f"    FRAGILE: {n_f}\n")
            if n_r > 0:
                combined_p = (1 / (N_RANDOM_TRIALS + 1)) ** n_r
                f.write(f"    Combined p: < {combined_p:.2e}\n")
                robust_deltas = [abs(r['delta']) for r in flips if r.get('robustness') == 'ROBUST']
                f.write(f"    Mean |Δ| of robust: {np.mean(robust_deltas):.3f}\n")
            f.write("\n")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
