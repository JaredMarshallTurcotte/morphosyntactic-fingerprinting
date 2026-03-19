"""
Soft accuracy analysis for the MorphTagTransformer ensemble.
Computes AUC-ROC, calibration, and margin-aware accuracy on held-out validation data.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.constants import MORPH_CODE_DIM, DEPREL_TO_IDX
from data.feature_extractor import SENTENCE_GRAMMAR_PROFILE_DIM
from models.morph_transformer import MorphTagTransformer

# Reuse encoding from train_ensemble
from analysis.train_ensemble import (
    encode_morph_code, encode_sentence, LazyGrammarDataset,
    stratified_split, load_corpus,
)

ENSEMBLE_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
ENSEMBLE_DIR = ROOT / "trained_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
VAL_RATIO = 0.1


def load_model(seed):
    path = ENSEMBLE_DIR / f"grammar_ensemble_seed{seed}.pt"
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model = MorphTagTransformer(
        morph_dim=MORPH_CODE_DIM,
        deprel_vocab_size=len(DEPREL_TO_IDX),
        sentence_profile_dim=SENTENCE_GRAMMAR_PROFILE_DIM,
        num_classes=2,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def get_val_predictions(model, val_loader):
    """Get P(Jewish) for every validation sample."""
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            morph = batch['morph_seq'].to(DEVICE)
            deprel = batch['deprel_seq'].to(DEVICE)
            mask = batch['morph_seq_mask'].to(DEVICE)
            gp = batch['sentence_grammar_profile'].to(DEVICE)
            target = batch['label']

            logits = model(morph, deprel, mask, gp)
            probs = F.softmax(logits, dim=1)  # [batch, 2]
            p_jewish = probs[:, 0].cpu().numpy()  # class 0 = jewish

            all_probs.extend(p_jewish)
            all_labels.extend(target.numpy())

    return np.array(all_probs), np.array(all_labels)


def compute_auc_roc(probs, labels):
    """Manual AUC-ROC computation (no sklearn dependency)."""
    # For AUC, we want P(positive class). Jewish = 0, so P(Jewish) is our score
    # and label=0 is the positive class.
    # Sort by descending probability
    sorted_indices = np.argsort(-probs)
    sorted_labels = labels[sorted_indices]
    sorted_probs = probs[sorted_indices]

    n_pos = (labels == 0).sum()
    n_neg = (labels == 1).sum()

    # Compute TPR and FPR at each threshold
    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0

    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 0:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

    return auc, np.array(fpr_list), np.array(tpr_list)


def compute_calibration(probs, labels, n_bins=10):
    """Compute calibration: in each probability bin, what fraction is actually Jewish?"""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_actual = []
    bin_predicted = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_predicted.append(probs[mask].mean())
        bin_actual.append((labels[mask] == 0).mean())  # fraction actually Jewish
        bin_counts.append(mask.sum())

    # ECE (Expected Calibration Error)
    total = sum(bin_counts)
    ece = sum(abs(a - p) * c / total for a, p, c in zip(bin_actual, bin_predicted, bin_counts))

    return bin_centers, bin_actual, bin_predicted, bin_counts, ece


def compute_margin_accuracy(probs, labels, margins):
    """Accuracy considering only predictions above a confidence margin."""
    results = {}
    for margin in margins:
        # Prediction: Jewish if P(J) >= 0.5, Gentile otherwise
        predicted = (probs >= 0.5).astype(int)  # 1 = Jewish prediction
        predicted = 1 - predicted  # flip: 0 = Jewish label

        confident = np.abs(probs - 0.5) >= margin
        if confident.sum() == 0:
            results[margin] = (0, 0, 0.0)
            continue
        correct = (predicted[confident] == labels[confident])
        acc = correct.mean()
        results[margin] = (int(confident.sum()), len(labels), acc)

    return results


def main():
    # Parse corpus once
    sentences, labels = load_corpus()

    # We'll use a held-out seed (99) for a clean evaluation split that
    # none of the models were trained on
    print("\n  Creating held-out evaluation split (seed=99)...")
    eval_train_idx, eval_val_idx = stratified_split(labels, val_ratio=0.2, seed=99)
    # Use the 20% split as our eval set
    eval_sents = [sentences[i] for i in eval_val_idx]
    eval_labels = labels[eval_val_idx]

    n_jewish = (eval_labels == 0).sum()
    n_gentile = (eval_labels == 1).sum()
    print(f"  Eval set: {len(eval_labels):,} sentences (Jewish: {n_jewish:,}, Gentile: {n_gentile:,})")

    eval_ds = LazyGrammarDataset(eval_sents, eval_labels)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Collect ensemble predictions
    print("\n  Running ensemble predictions...")
    all_model_probs = []

    for seed in ENSEMBLE_SEEDS:
        print(f"    Seed {seed}...", end=" ", flush=True)
        model = load_model(seed)
        probs, true_labels = get_val_predictions(model, eval_loader)
        all_model_probs.append(probs)
        hard_acc = ((probs >= 0.5) == (true_labels == 0)).mean()
        print(f"hard acc = {hard_acc:.4f}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Ensemble average
    all_model_probs = np.array(all_model_probs)  # [10, N]
    ensemble_probs = all_model_probs.mean(axis=0)  # [N]
    ensemble_std = all_model_probs.std(axis=0)  # [N]

    # Hard accuracy
    ensemble_pred = (ensemble_probs >= 0.5).astype(int)
    ensemble_correct = (ensemble_pred == (true_labels == 0).astype(int))
    hard_acc = ensemble_correct.mean()

    print(f"\n  Ensemble hard accuracy (threshold=0.5): {hard_acc:.4f}")

    # === AUC-ROC ===
    auc, fpr, tpr = compute_auc_roc(ensemble_probs, true_labels)
    print(f"  AUC-ROC: {auc:.4f}")

    # === Calibration ===
    bin_centers, bin_actual, bin_predicted, bin_counts, ece = compute_calibration(
        ensemble_probs, true_labels, n_bins=10
    )
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")

    print("\n  Calibration table:")
    print(f"  {'Bin':>12} {'Predicted':>10} {'Actual':>10} {'Count':>8}")
    print(f"  {'-'*42}")
    for c, a, p, n in zip(bin_centers, bin_actual, bin_predicted, bin_counts):
        print(f"  {c:>10.1%}   {p:>9.3f}   {a:>9.3f}   {n:>6d}")

    # === Margin-aware accuracy ===
    margins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    margin_results = compute_margin_accuracy(ensemble_probs, true_labels, margins)

    print("\n  Margin-aware accuracy:")
    print(f"  {'Margin':>8} {'Confident':>10} {'Total':>8} {'Coverage':>10} {'Accuracy':>10}")
    print(f"  {'-'*50}")
    for margin in margins:
        n_conf, n_total, acc = margin_results[margin]
        coverage = n_conf / n_total if n_total > 0 else 0
        print(f"  {margin:>8.2f} {n_conf:>10,} {n_total:>8,} {coverage:>9.1%} {acc:>9.4f}")

    # === Per-model AUC ===
    print("\n  Per-model AUC-ROC:")
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        auc_i, _, _ = compute_auc_roc(all_model_probs[i], true_labels)
        print(f"    Seed {seed}: {auc_i:.4f}")

    # === Brier Score ===
    # Brier score: mean squared error of probabilistic predictions
    jewish_indicator = (true_labels == 0).astype(float)
    brier = np.mean((ensemble_probs - jewish_indicator) ** 2)
    print(f"\n  Brier Score: {brier:.4f} (lower is better; 0.25 = random)")

    # === Summary for paper ===
    print("\n" + "=" * 60)
    print("  SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"  Evaluation set:      {len(true_labels):,} sentences (20% held-out, seed=99)")
    print(f"  Hard accuracy:       {hard_acc:.1%}")
    print(f"  AUC-ROC:             {auc:.4f}")
    print(f"  Brier Score:         {brier:.4f}")
    print(f"  ECE:                 {ece:.4f}")

    # Key margin thresholds for the paper
    for margin in [0.10, 0.20]:
        n_conf, n_total, acc = margin_results[margin]
        coverage = n_conf / n_total
        print(f"  Acc @ margin {margin}: {acc:.1%} (coverage: {coverage:.1%})")

    print()


if __name__ == "__main__":
    main()
