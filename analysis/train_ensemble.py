"""
Deep Ensemble training: 10 MorphTagTransformer models with different seeds.
Parses corpus once, then trains each seed sequentially.

Usage:
    python train_ensemble.py
    (or double-click run_training.bat)
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ── path setup ──────────────────────────────────────────────────────────
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

# ── config ──────────────────────────────────────────────────────────────
CORPUS_ROOT = ROOT / "Corpus"
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 0.01
VAL_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENSEMBLE_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
ENSEMBLE_DIR = ROOT / "trained_models"

BAR_WIDTH = 40


# ── progress bar ────────────────────────────────────────────────────────
def progress_bar(current, total, width=BAR_WIDTH, prefix="", suffix=""):
    frac = current / total if total > 0 else 1.0
    filled = int(width * frac)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = frac * 100
    line = f"\r  {prefix} [{bar}] {pct:5.1f}% {suffix}"
    sys.stdout.write(line)
    sys.stdout.flush()


def print_header(text):
    w = 60
    print()
    print("=" * w)
    print(f"  {text}")
    print("=" * w)


def print_divider():
    print("-" * 60)


# ── feature encoding ────────────────────────────────────────────────────
def encode_morph_code(proiel_code):
    vec = []
    for pos_idx, vocab in enumerate(PROIEL_POSITION_VOCABS):
        one_hot = [0.0] * len(vocab)
        if pos_idx < len(proiel_code):
            char = proiel_code[pos_idx]
            if char in vocab:
                one_hot[vocab.index(char)] = 1.0
        vec.extend(one_hot)
    return vec


def encode_sentence(sentence):
    tokens = sentence.word_tokens[:MAX_MORPH_SEQ_LEN]
    morph_seq = np.zeros((MAX_MORPH_SEQ_LEN, MORPH_CODE_DIM), dtype=np.float32)
    deprel_seq = np.full(MAX_MORPH_SEQ_LEN, DEPREL_PAD_IDX, dtype=np.int64)
    mask = np.zeros(MAX_MORPH_SEQ_LEN, dtype=np.bool_)

    for i, token in enumerate(tokens):
        morph = MorphologyNormalizer.conllu_to_features(token)
        code = MorphologyNormalizer.features_to_proiel(morph)
        morph_seq[i] = encode_morph_code(code)
        deprel = token.deprel.split(':')[0] if token.deprel else 'other'
        deprel_seq[i] = DEPREL_TO_IDX.get(deprel, DEPREL_TO_IDX['other'])
        mask[i] = True

    grammar_profile = extract_sentence_grammar_profile(sentence)
    return morph_seq, deprel_seq, mask, grammar_profile


# ── lazy dataset ────────────────────────────────────────────────────────
class LazyGrammarDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        morph_seq, deprel_seq, mask, grammar_profile = encode_sentence(sent)
        return {
            'morph_seq': torch.from_numpy(morph_seq),
            'deprel_seq': torch.from_numpy(deprel_seq),
            'morph_seq_mask': torch.from_numpy(mask),
            'sentence_grammar_profile': torch.from_numpy(
                np.array(grammar_profile, dtype=np.float32)
            ),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── corpus parsing ─────────────────────────────────────────────────────
def ensure_corpus_extracted():
    """Extract training_corpus.tar.gz if Corpus/jewish/ and Corpus/gentile/ don't exist."""
    jewish_dir = CORPUS_ROOT / "jewish"
    gentile_dir = CORPUS_ROOT / "gentile"
    if jewish_dir.exists() and gentile_dir.exists():
        return
    archive = CORPUS_ROOT / "training_corpus.tar.gz"
    if archive.exists():
        import tarfile
        print(f"  Extracting {archive.name}...")
        with tarfile.open(archive, 'r:gz') as tf:
            tf.extractall(path=CORPUS_ROOT)
        print(f"  Extracted to {CORPUS_ROOT}")
    else:
        print(f"  ERROR: Neither extracted corpus dirs nor {archive} found.")
        print(f"  Place jewish/ and gentile/ CoNLL-U directories in {CORPUS_ROOT}/")
        sys.exit(1)


def load_corpus():
    print_header("PARSING CORPUS (once for all seeds)")
    print(f"  Source: {CORPUS_ROOT}")
    ensure_corpus_extracted()
    print(f"  Loading CoNLL-U files...")
    sys.stdout.flush()

    t0 = time.time()
    parser = ConlluParser()
    corpora = parser.parse_tagged_corpus(str(CORPUS_ROOT))
    parse_time = time.time() - t0
    print(f"  Parsed in {parse_time:.1f}s")

    sentences = []
    labels = []
    for corpus in corpora:
        label = 0 if corpus.tradition == "jewish" else 1
        for sent in corpus.sentences:
            sentences.append(sent)
            labels.append(label)

    labels = np.array(labels, dtype=np.int64)
    n_jewish = int((labels == 0).sum())
    n_gentile = int((labels == 1).sum())
    print(f"  Found {len(corpora)} traditions, {len(sentences):,} total sentences")
    print(f"  Jewish: {n_jewish:,}  |  Gentile: {n_gentile:,}")

    return sentences, labels


# ── train/val split ────────────────────────────────────────────────────
def stratified_split(labels, val_ratio, seed):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_ratio))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


# ── single model training ──────────────────────────────────────────────
def train_one(seed, sentences, labels, model_idx, total_models):
    output_path = ENSEMBLE_DIR / f"grammar_ensemble_seed{seed}.pt"

    # Skip if already trained
    if output_path.exists():
        ckpt = torch.load(str(output_path), map_location="cpu", weights_only=False)
        print(f"\n  Seed {seed} already trained (val_acc={ckpt['val_acc']:.4f}), skipping.")
        return ckpt['val_acc']

    torch.manual_seed(seed)
    np.random.seed(seed)

    print_header(f"MODEL {model_idx}/{total_models}  (seed={seed})")

    # Split (different seed = different split)
    train_idx, val_idx = stratified_split(labels, VAL_RATIO, seed)
    print(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}")

    train_sents = [sentences[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_sents = [sentences[i] for i in val_idx]
    val_labels = labels[val_idx]

    train_ds = LazyGrammarDataset(train_sents, train_labels)
    val_ds = LazyGrammarDataset(val_sents, val_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)

    # Model
    model = MorphTagTransformer(
        morph_dim=MORPH_CODE_DIM,
        deprel_vocab_size=len(DEPREL_TO_IDX),
        sentence_profile_dim=SENTENCE_GRAMMAR_PROFILE_DIM,
        num_classes=2,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: {param_count:,} parameters  |  Device: {DEVICE}")
    print(f"  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}  |  LR: {LR}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    total_train_steps = len(train_loader)
    total_val_steps = len(val_loader)

    print()
    print(f"  {'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Time':<8} {'Status'}")
    print_divider()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            morph = batch['morph_seq'].to(DEVICE)
            deprel = batch['deprel_seq'].to(DEVICE)
            mask = batch['morph_seq_mask'].to(DEVICE)
            gp = batch['sentence_grammar_profile'].to(DEVICE)
            target = batch['label'].to(DEVICE)

            logits = model(morph, deprel, mask, gp)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * target.size(0)
            train_correct += (logits.argmax(1) == target).sum().item()
            train_total += target.size(0)

            progress_bar(step, total_train_steps,
                         prefix=f"Ep {epoch:2d}/{EPOCHS}",
                         suffix=f"batch {step}/{total_train_steps}   ")

        train_time = time.time() - t0
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for vs, batch in enumerate(val_loader, 1):
                morph = batch['morph_seq'].to(DEVICE)
                deprel = batch['deprel_seq'].to(DEVICE)
                mask = batch['morph_seq_mask'].to(DEVICE)
                gp = batch['sentence_grammar_profile'].to(DEVICE)
                target = batch['label'].to(DEVICE)

                logits = model(morph, deprel, mask, gp)
                loss = criterion(logits, target)

                val_loss += loss.item() * target.size(0)
                val_correct += (logits.argmax(1) == target).sum().item()
                val_total += target.size(0)

                progress_bar(vs, total_val_steps,
                             prefix=f"Ep {epoch:2d}/{EPOCHS}",
                             suffix=f"validating {vs}/{total_val_steps}   ")

        val_loss /= val_total
        val_acc = val_correct / val_total

        saved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'seed': seed,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, str(output_path))
            saved = " * BEST *"

        sys.stdout.write("\r" + " " * 100 + "\r")
        print(f"  {epoch:<8d} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f} {train_time:<8.1f}{saved}")
        sys.stdout.flush()

    print(f"\n  Seed {seed} done. Best val_acc: {best_val_acc:.4f}")
    print(f"  Saved to: {output_path}")

    return best_val_acc


# ── main ────────────────────────────────────────────────────────────────
def main():
    ENSEMBLE_DIR.mkdir(exist_ok=True)

    # Parse corpus once
    sentences, labels = load_corpus()

    # Train each seed
    results = []
    total_t0 = time.time()

    for i, seed in enumerate(ENSEMBLE_SEEDS, 1):
        val_acc = train_one(seed, sentences, labels, i, len(ENSEMBLE_SEEDS))
        results.append((seed, val_acc))

    total_time = time.time() - total_t0

    # Summary
    print_header("DEEP ENSEMBLE COMPLETE")
    print(f"  Models: {len(ENSEMBLE_SEEDS)}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print()
    print(f"  {'Seed':<8} {'Val Acc':<12} {'File'}")
    print_divider()
    accs = []
    for seed, acc in results:
        fname = f"grammar_ensemble_seed{seed}.pt"
        print(f"  {seed:<8d} {acc:<12.4f} {fname}")
        accs.append(acc)

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print_divider()
    print(f"  Mean: {mean_acc:.4f} +/- {std_acc:.4f}")
    print()
    print(f"  Ensemble models saved to: {ENSEMBLE_DIR}/")
    print(f"  Use all 10 models at inference time — average softmax outputs")
    print(f"  and use variance as uncertainty signal.")
    print()


if __name__ == "__main__":
    main()
