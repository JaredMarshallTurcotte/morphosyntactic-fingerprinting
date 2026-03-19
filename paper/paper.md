# Morphosyntactic Fingerprinting of Editorial Layers in the Greek New Testament

**Jared Marshall Turcotte**
jmtovw@gmail.com

---

## Abstract

I present a grammar only deep learning approach to textual stylometry for the Greek New Testament corpus. A transformer based approach without any access to lexical or surface features, such as word usage. Operating purely on morphosyntactic features, we are able to achieve 80.7% hard accuracy distinguishing between Jewish and Gentile tradition Greek writing (AUC-ROC: 0.892; 96.3% accuracy on confident predictions at ±0.30 margin). Using a 10 model deep ensemble with uncertainty quantification, all 27 New Testament books have been analyzed at both verse and clause granularity. Along with the main methodology, I have developed a redaction detection system that seeks to unveil textual redaction that may permeate the New Testament documents. By stripping lemmas we hypothesize to be editorial insertions, and measuring the resulting shift in the classification, we may be able to uncover an underlying textual substrate which has been hidden for millennia. This methodology has been tested on three independent marker sets, those being, Christological/patriarchal, Catholic institutional, and expanded theological. The approach identifies 21 Jewish → Gentile tradition flips deemed robust, validated by random ablation (0/100 random flips each). The reverse direction (Gentile → Jewish) was explicitly tested across all three marker sets and produces only 2 robust flips, a 10.5:1 asymmetry. The results of the tradition flips are actually quite remarkable, resolving various issues including the awkward Greek within James 2:1, the centuries old debate surrounding post-apostasy repentance in Hebrews 6, the circular logic of the Philippians 2 kenosis hymn ("in the form of God... did not grasp at equality with God"), and the gnosis substrate in Philippians 3:8 where stripping Christological tokens reveals "the surpassing value of knowledge" as the complete thought. Broadly speaking the tradition flips which are identified as robust, especially those containing the highest Delta changes, appear to strongly correlate with a more coherent reading.

---

## 1. Introduction

Stylometry for the New Testament documents has quite an extensive history, from F. C. Baur's Tübingen school to modern redaction criticism. The central question which permeates all of these endeavours can be distilled in this: do New Testament texts preserve earlier documentary layers that were later edited to fit emerging orthodox theology? A related question concerns origin. Could the canonical documents be Judaized redactions of originally Gentile writings? Past approaches to these questions have relied almost entirely on surface features such as word count and other forms of vocabulary analysis, theological argumentation, and manuscript comparison. These methods produce compelling hypotheses, but they are deeply vulnerable to circular reasoning.

Computational stylometry has been applied to biblical authorship questions since de Morgan's 1851 analysis of the Pauline epistles (Morton, 1978; Kenny, 1986). More recent work has used machine learning for Hebrew Bible source criticism (Faigenbaum-Golovin et al., 2023). These approaches invariably rely on surface features, word frequencies, vocabulary distributions, and other things which are easily confounded by topic or target audience. Lexical or surface-based approaches are inevitably bound to the circular nature of what we think the original author wrote, which ultimately ends up dictating the classification.

We must cease grasping at surface-level features and change course entirely. That is the motivation behind morphosyntactic fingerprinting. By training a classifier that sees only grammatical structure, which is composed of part-of-speech sequences, morphological inflection patterns, dependency relations, and syntactic complexity measures, we isolate the underlying native grammar or "thought pattern" of a writer independent of semantic content. A text can discuss Jewish themes using Gentile grammatical patterns, or the inverse. If a passage classified as Jewish by its grammar flips to Gentile when editorial markers are stripped, the remaining grammatical substrate may preserve an earlier tradition.

The architecture of morphosyntactic fingerprinting allows us to develop our corpora on texts which are not our target text. This allows for two astounding benefits: first, it abolishes the issue of circularity, and second, it allows us to build classification models across far larger corpora than previously possible. The New Testament never appears in the training data. Target texts are scored against grammatical profiles learned from independent sources, as opposed to being compared against each other or against assumptions about their own authorship. We know who Josephus was. We know who Plutarch was. We can model what their grammar looks like.

The primary contributions of this work are:

1. A **grammar-only tradition classifier** (MorphTagTransformer, 358K parameters) achieving 80.7% hard accuracy, AUC-ROC 0.892, 96.3% on confident predictions. No lexical access whatsoever.
2. A **10-model deep ensemble** with uncertainty quantification via inter-model disagreement.
3. **Verse and clause-level scoring** of all 27 New Testament books.
4. An **automated redaction detection system** that strips editorial markers, re-scores, identifies tradition flips, and validates against random ablation.
5. **Open-source code and data** for full reproducibility.

## 2. Related Work

Prior implementations of stylometry on biblical texts have relied on word frequencies (Morton, 1978; Kenny, 1986), vocabulary distributions (Faigenbaum-Golovin et al., 2023), and function word ratios. Eder et al. (2023) demonstrated that morphosyntactic annotation is capable of providing topic-agnostic stylometric features, and DeepMind's Ithaca (Assael et al., 2022) demonstrated deep learning on ancient Greek inscriptions, however it was applied for the purposes of dating and geographic attribution, not tradition classification. To the best of my knowledge, this is the first application of grammar-only tradition classification and automated redaction detection via ensemble uncertainty in biblical text analysis.

## 3. Data

### Training Corpus

The training corpora consists of 125,021 CoNLL-U annotated Ancient Greek sentences, which have been divided into two tradition classes, and balanced at roughly 62,900 sentences each:

**Jewish tradition** (62,886 sentences, 44 sources): Philo Judaeus (23,851), Flavius Josephus (20,228), Clementine Pseudepigrapha (5,706), Shepherd of Hermas (1,432), Sibylline Oracles (1,265), Testament of Solomon (1,220), intertestamental literature (Testaments of the Twelve Patriarchs, Testament of Abraham, Testament of Job, 1 Enoch, Jubilees), Jewish apocalyptic texts (2/3 Baruch, 4 Ezra, Apocalypse of Elijah), apocryphal acts (Acts of John, Paul, Peter), and early Jewish-Christian writings (Epistle of Barnabas, Didache, 2 Clement).

**Gentile tradition** (62,135 sentences, 9 sources): Plutarch (11,878), Polybius (10,371), Dio Chrysostom (10,102), Epictetus (9,385), Claudius Aelianus (7,695), Flavius Arrianus (5,948), Aelius Aristides (4,406), Marcus Aurelius (1,420), Gaius Musonius Rufus (930).

The corpora themselves have been deliberately balanced by sentence count and genre in order to prevent any biases. The Jewish corpora has greater source diversity, with 44 authors as opposed to nine in the Gentile corpora; however, it is dominated by two major authors, Philo and Josephus, who account for ~70% of the corpus. The Gentile corpus has fewer sources but a more even distribution. The median sentence length across the whole corpus is 17 tokens.

### New Testament Corpus

The 27 New Testament documents have been provided as CoNLL-U files with Universal Dependencies, morphology and syntax. Since the training data has natural sentences averaging 17 tokens, I split the New Testament text into clauses at the Greek punctuation boundaries (`.` `·` `;`) with a minimum of three tokens per clause and a maximum of 64, which yields units averaging 14 tokens.

## 4. Model Architecture

### MorphTagTransformer

The MorphTagTransformer is a small transformer containing 358K parameters, that takes four input streams, and produces a binary tradition classification.

#### Input Representation

**Morphological Sequence.** Each token is encoded as a PROIEL 10-position morphological code covering part-of-speech, person, number, tense, mood, voice, gender, case, degree, and definiteness. Each position is then one-hot encoded using position specific vocabulary, which results in a 62 dimensional vector for each token.

**Table 1: PROIEL morphological code positions and their one-hot encoding dimensions. Total: 62 dimensions per token.**

| Position | Feature        | Dim | Values                             |
| -------- | -------------- | --- | ---------------------------------- |
| 1        | Part of Speech | 12  | N, V, A, D, C, R, P, M, I, G, X, - |
| 2        | Person         | 4   | 1, 2, 3, -                         |
| 3        | Number         | 4   | s, p, d, -                         |
| 4        | Tense          | 9   | p, i, r, s, a, u, l, f, -          |
| 5        | Mood           | 9   | i, s, m, o, n, p, d, g, -          |
| 6        | Voice          | 5   | a, m, p, e, -                      |
| 7        | Gender         | 5   | m, f, n, c, -                      |
| 8        | Case           | 7   | n, a, g, d, v, o, -                |
| 9        | Degree         | 4   | p, c, s, -                         |
| 10       | Definiteness   | 3   | w, s, -                            |

**Dependency Relation Sequence.** Each token's syntactic role (e.g., `nsubj`, `obj`, `obl`, `nmod`) is mapped to a 21-category vocabulary, and then embedded into a 16-dimensional learned vector.

**Grammar Profile.** A 391-dimensional sentence-level feature vector capturing:

- POS, dependency relation, case, tense, mood, and voice distributions (58 features)
- Morphological n-gram frequencies (305 features): POS bigrams, trigrams, and morphological pattern sequences
- Stylometric features (10 features): mean token length, type-token ratio, function word ratio, etc.
- Syntactic complexity features (18 features): tree depth, genitive chain depth, clause embedding depth, coordination patterns

#### Architecture

Each token gets a morphological one-hot vector (62 dimensions) concatenated with its dependency relation embedding (16 dimensions), so 78 total. That gets projected down to 128 dimensions through a linear layer, then summed with learned positional encodings. From there it runs through a 2-layer pre-norm transformer encoder, 4 attention heads, GELU activations, 256-dimensional feed-forward layers.

After the transformer, I mean-pool over the non-padding tokens, then concatenate that with the 391-dimensional grammar profile. So the full representation is 519 dimensions. That feeds into a 3-layer MLP (519 → 128 → 64 → 2) with ReLU and dropout at 0.1, which gives the final class logits.

### Deep Ensemble

I train 10 instances of the MorphTagTransformer with different random seeds (42--51), each with independently randomized train/validation splits (90/10 stratified). At inference, all 10 models produce softmax probability distributions, which are averaged for the ensemble prediction P̄(J). The inter-model standard deviation σ serves as an uncertainty estimate:

*P̄(J) = (1/10) Σₖ Pₖ(J),   σ = √[(1/10) Σₖ (Pₖ(J) − P̄(J))²]*

I define four σ confidence bands: **low** (σ < 0.08), **moderate** (0.08 ≤ σ < 0.15), **high** (0.15 ≤ σ < 0.25), and **EXTREME** (σ ≥ 0.25). Going from 5 to 10 models made a significant difference. The uncertainty estimates are noticeably more stable and the overall accuracy improved across every metric.

### Training Details

Each model trains for 20 epochs with batch size 256, AdamW optimizer (learning rate 10⁻³, weight decay 0.01), and OneCycleLR cosine schedule with CrossEntropyLoss. Best checkpoint by validation accuracy is saved. Individual model validation accuracies range from 69.4% to 70.2% (mean: 69.8%) across all 10 seeds.

### Soft Accuracy Analysis

The ~70% per-model validation accuracy sounds underwhelming on its own, but it's misleading. A hard 0.5 threshold treats a prediction of P(J)=0.51 on a Gentile text the same as P(J)=0.95, but ancient Greek exists on a genuine tradition continuum. A Hellenistic Jewish author educated in Greek philosophical schools would naturally produce grammar near the boundary at times, and the model scoring them there is arguably correct. To get a better picture, I evaluated the 10-model ensemble on a 20% held-out set (25,004 sentences, seed=99, disjoint from all training and validation splits).

**Table 1b: Ensemble evaluation metrics on held-out data.**

| Metric                        | Value                 |
| ----------------------------- | --------------------- |
| Hard accuracy (threshold=0.5) | 80.7%                 |
| AUC-ROC                       | 0.892                 |
| Brier Score                   | 0.140 (0.25 = random) |
| Expected Calibration Error    | 0.058                 |

The AUC-ROC of 0.892 is the number of foremost importance. This is really what demonstrates how well the model is separating the two classes across all thresholds regardless of where you place the boundary. The Brier score of 0.140 which is well below the 0.25 baseline (which would deem the results as effectively random) confirms that the model is producing meaningful probabilities and not simply guessing.

**Margin-aware accuracy.** The ensemble is already well above 80% accuracy when accounting for all data. Accuracy is improved significantly when we filter out predictions which exist near the decision boundary.

**Table 1c: Accuracy by confidence margin. Margin = minimum distance from 0.5 required to count as a confident prediction.**

| Margin | Confident | Coverage | Accuracy |
| ------ | --------- | -------- | -------- |
| 0.00   | 25,004    | 100.0%   | 80.7%    |
| 0.05   | 22,087    | 88.3%    | 84.1%    |
| 0.10   | 19,352    | 77.4%    | 87.1%    |
| 0.15   | 16,710    | 66.8%    | 89.7%    |
| 0.20   | 14,116    | 56.5%    | 92.1%    |
| 0.25   | 11,496    | 46.0%    | 94.5%    |
| 0.30   | 9,020     | 36.1%    | 96.3%    |

At ±0.10, just throwing out the 23% of predictions closest to the boundary, accuracy is already 87.1%. At ±0.30, it's 96.3% on the remaining 36% of data. The boundary cases are genuinely ambiguous.

**Calibration.** ECE = 0.058. The predicted probabilities track actual frequencies closely, so if the model says P(J) = 0.8, you can trust that number. The raw outputs mean what they say.

**Ensemble scaling.** The 10-model ensemble improves on all prior configurations (5-model: 79.9% hard accuracy, AUC-ROC 0.885, Brier 0.143) across every metric. Per-model AUC-ROC ranges from 0.845 to 0.878 across the ten seeds, with the ensemble hitting 0.892.

## 5. Methods

### Clause-Level Analysis

For each of the 27 New Testament books:

1. Parse the CoNLL-U file to extract tokens with morphological annotations
2. Split tokens into natural clauses at Greek punctuation boundaries
3. Encode each clause via the input representation described above
4. Score through the 10-model ensemble

### Redaction Detection

The redaction detection methodology tests a specific question: do Jewish-classified clauses contain non-Jewish grammatical substrate that's being masked by editorial insertions? For each clause with P̄(J) ≥ 0.5, I define categories of suspected editorial markers, identified by lemma in the CoNLL-U annotation. I test this across three independent marker sets, and in both directions (J→G and G→J).

**Table 2: Three independent editorial marker sets. Matching is by lemma, automatically capturing all inflected forms.**

| Set                               | Category         | Lemmas                                                  |
| --------------------------------- | ---------------- | ------------------------------------------------------- |
| **A: Christological/Patriarchal** | Christological   | *Ἰησοῦς, χριστός*                                       |
|                                   | Kyrios           | *κύριος*                                                |
|                                   | Patriarchal      | *Μωϋσῆς, Ἀβραάμ, Δαυίδ, Ἰσραήλ, Ἰακώβ, Ἰσαάκ*           |
|                                   | Scripture        | *γράφω*                                                 |
|                                   | Prophetic        | *προφήτης, Ἠσαΐας, Ἰερεμίας, Ἠλίας*                     |
|                                   | Torah/Law        | *νόμος, ἐντολή, περιτομή*                               |
| **B: Catholic**                   | Ecclesiological  | *ἐκκλησία, ἐπίσκοπος, πρεσβύτερος, διάκονος, ἀπόστολος* |
|                                   | Sacramental      | *βαπτίζω, βάπτισμα*                                     |
|                                   | Creedal          | *ἀνάστασις, σταυρός, σταυρόω*                           |
|                                   | Hamartiology     | *ἁμαρτία, μετάνοια, μετανοέω*                           |
|                                   | Pneumatological  | *πνεῦμα*                                                |
|                                   | Eschatological   | *παρουσία, βασιλεία*                                    |
| **C: Expanded**                   | Divine reference | *θεός, πατήρ*                                           |
|                                   | Holiness         | *ἅγιος, δόξα*                                           |
|                                   | Soteriology      | *σωτήρ, σωτηρία, εὐαγγέλιον*                            |
|                                   | Faith            | *πίστις*                                                |

The three sets share no lemmas. If the methodology is detecting a real signal rather than an artifact of a particular category choice, the results should be consistent across all three.

#### Stripping Procedure

When one of the target tokens is removed, its syntactic dependents can become orphaned. In order to solve this issue, I cascade the stripping to dependents connected via `det`, `amod`, `case`, `nmod`, `flat`, and `flat:name`, iterating until stable. So stripping *Ἰησοῦ* from *τοῦ κυρίου ἡμῶν Ἰησοῦ Χριστοῦ* cascades to remove the entire phrase, article, possessive pronoun etc.

#### Re-scoring

The remaining tokens (minimum 3 non-punctuation required) are re-encoded and re-scored through the ensemble. I compute:

- **ΔP(J) = P̄(J)\_original − P̄(J)\_stripped**: positive values indicate movement toward Gentile
- **J→G tradition flip**: a clause where P̄(J)\_original ≥ 0.5 and P̄(J)\_stripped < 0.5
- **G→J tradition flip**: a clause where P̄(J)\_original < 0.5 and P̄(J)\_stripped ≥ 0.5

I run both an all-strip pass (removing all editorial categories simultaneously) and per-category passes (one category at a time) for ablation. Each marker set is tested in both directions. For each flip, 100 random ablation trials (stripping the same number of non-editorial tokens with identical cascade logic) determine whether the flip is ROBUST (0/100 random), RARE (1--5/100), or FRAGILE (>5/100).

## 6. Results

### Ensemble Classification

Table 3 shows a selection of per-book results at clause granularity. Full results for all 27 books are in the supplementary materials.

**Table 3: Selected book-level ensemble results at clause granularity.**

| Book          | Mean P̄(J) | ± std | Label  |
| ------------- | ---------- | ----- | ------ |
| 1 Corinthians | 0.663      | 0.131 | Jewish |
| 1 Peter       | 0.733      | 0.121 | Jewish |
| Romans        | 0.743      | 0.112 | Jewish |
| Titus         | 0.755      | 0.112 | Jewish |
| 2 Timothy     | 0.798      | 0.110 | Jewish |
| Acts          | 0.811      | 0.095 | Jewish |
| 2 Peter       | 0.815      | 0.085 | Jewish |
| Philemon      | 0.859      | 0.098 | Jewish |

Every single NT book scores Jewish. That is expected. These are texts that have been transmitted, copied, and edited within a Jewish-Christian scribal tradition for centuries. The question is not whether they score Jewish on the whole, but whether specific clauses contain Gentile substrate beneath the editorial surface.

### Redaction Detection

#### J→G Results: Three Marker Sets

**Set A (Christological/Patriarchal):** Of 299 scorable Jewish clauses, 24 flip (8.0%). **12 are robust** (0/100 random, combined p < 8.87 × 10⁻²⁵). The "Known" Pauline letters show the highest flip rate at 12.9%. Torah/Law markers produce the highest per-category flip rate (19.0%), while Christological markers produce the most flips by count (17) simply because they appear so frequently.

The five largest robust flips:

- **1 Thessalonians 5:9** (Δ = +0.486): Stripping *τοῦ κυρίου ἡμῶν Ἰησοῦ Χριστοῦ* yields "God has not destined us for wrath but for obtaining salvation through..." (P(J): 0.806 → 0.320). Soteriology with no Christological attribution. Just "obtaining salvation" as an abstract act.
- **Philippians 3:8** (Δ = +0.471): Stripping *Χριστοῦ Ἰησοῦ τοῦ κυρίου μου* reveals "I count all things loss for the surpassing of *knowledge*" (P(J): 0.773 → 0.302). A gnosis formula. The knowledge itself is the goal, not knowledge *of Christ Jesus my Lord*.
- **James 2:1** (Δ = +0.453): Stripping *τοῦ κυρίου ἡμῶν Ἰησοῦ Χριστοῦ τῆς* yields "Brothers, do not hold the faith of glory with partiality" (P(J): 0.848 → 0.395). πίστιν δόξης, "faith of glory," reads like an abstract hypostatic concept without the Christological attribution. The Greek is also considerably less awkward.
- **Romans 9:3--5** (Δ = +0.421): Stripping Christological tokens from this passage about Israel's privileges (P(J): 0.865 → 0.444). What remains is the enumeration of covenantal prerogatives (adoption, glory, covenants, law, worship, promises) without the Christological doxology that caps it.
- **2 Timothy 2:3** (Δ = +0.262): Stripping *Χριστοῦ Ἰησοῦ* from "suffer hardship as a good soldier of Christ Jesus" (P(J): 0.753 → 0.491). A good soldier without a named commander.

Seven more flips are robust with smaller deltas (1 John 5:2, 2 John 1:6, Romans 4:9, Luke 7:4, Matthew 8:8, Matthew 28:5, John 5:1). Statistically real, but less revealing. Most involve removing a single *Ἰησοῦς* or *Κύριε* near the 0.5 boundary.

**Set B (Catholic):** Of 954 scorable Jewish clauses, 73 flip (7.7%). **3 are robust** (combined p < 9.71 × 10⁻⁷). Eschatological markers (βασιλεία, παρουσία) show the highest flip rate at 10.8%.

- **Hebrews 6:4--6** (Δ = +0.470, hamartiology + pneumatological): This passage has puzzled theologians for centuries because it seems to deny post-apostasy repentance, which contradicts mainstream soteriology. Strip the sin/repentance/spirit vocabulary and the contradiction vanishes. What remains is a text about enlightenment, tasting heavenly gifts, and the powers of the coming age. A coherent initiation text with no internal contradiction. The contradiction was *introduced* by the editorial layer.
- **Acts 4:8--10** (Δ = +0.468, ecclesiological + creedal): Stripping πρεσβύτεροι, Ἰσραήλ, the crucifixion reference (P(J): 0.712 → 0.243). What remains is a simple healing defence.
- **Acts 2:1--4** (Δ = +0.461, pneumatological): The Pentecost narrative without πνεῦμα. Tongues of fire and speaking in tongues, but without the Spirit attribution (P(J): 0.906 → 0.444).

**Set C (Expanded):** Of 1,630 scorable Jewish clauses, 159 flip (9.8%). **6 are robust** (combined p < 9.42 × 10⁻¹³). All six involve θεός/πατήρ. πίστις shows the highest per-category flip rate (12.4%).

- **Ephesians 6:5--8** (Δ = +0.604, divine_reference): The slave obedience instruction stripped of θεός and divine authorization (P(J): 0.906 → 0.301).
- **1 Peter 1:18--21** (Δ = +0.504, divine_reference + holiness + faith): "Redeemed from your futile way of life inherited from your ancestors." The phrase πατροπαραδότου (inherited from ancestors) describing a ματαίας ἀναστροφῆς (futile way of life) is remarkable. Calling ancestral tradition "futile" and requiring liberation from it reads more naturally on its own than it does wrapped in a Christian redemption narrative (P(J): 0.955 → 0.451).
- **Romans 2:15--16** (Δ = +0.474, divine_reference + soteriology): "The work of the law written on their hearts, their conscience bearing witness." Strip θεός and εὐαγγέλιον and what remains is a statement about innate moral knowledge. Conscience as internal witness, not divine judgment (P(J): 0.888 → 0.414).
- **Romans 3:21--22** (Δ = +0.426, divine_reference + faith): "Righteousness apart from law, witnessed by the law and prophets." Strip θεός and πίστις and it becomes an abstract universalist claim about righteousness that stands on its own (P(J): 0.826 → 0.400).
- **Acts 7:44--45** (Δ = +0.421, divine_reference): The tabernacle narrative stripped of patriarchal/divine references (P(J): 0.845 → 0.424).
- **Philippians 2:5--7** (Δ = +0.310, divine_reference): The kenosis hymn. "Who being in the form of God did not consider equality with God something to be grasped, but emptied himself." There is a well known logical tension here: if he is already in the form of God, why is equality with God something to grasp at? Strip θεός and the circularity disappears. What remains is a being who exists in a form, does not grasp at equality, and descends (P(J): 0.715 → 0.405).

#### G→J Results: Reverse Direction

I tested the same three marker sets in the G→J direction (stripping from Gentile-classified clauses). If the J→G signal were just syntactic damage from removing tokens, we would expect comparable numbers going the other way.

**Table 4: Bidirectional robustness results across all three marker sets.**

| Marker Set  | Direction | Total Flips | Robust | Combined p     |
| ----------- | --------- | ----------- | ------ | -------------- |
| A: Original | J→G       | 24          | 12     | < 8.87 × 10⁻²⁵ |
| A: Original | G→J       | 31          | 1      | < 9.90 × 10⁻³  |
| B: Catholic | J→G       | 73          | 3      | < 9.71 × 10⁻⁷  |
| B: Catholic | G→J       | 27          | 0      | n/a            |
| C: Expanded | J→G       | 159         | 6      | < 9.42 × 10⁻¹³ |
| C: Expanded | G→J       | 22          | 1      | < 9.90 × 10⁻³  |
| **Total**   | **J→G**   | **256**     | **21** |                |
| **Total**   | **G→J**   | **80**      | **2**  |                |

21 to 2. A 10.5:1 asymmetry. The two G→J robust flips are both marginal: Luke 18 (Δ = -0.243, "Jesus, son of David, have mercy on me," a Jewish plea formula that predictably scores more Jewish when the proper names are removed) and 1 Corinthians 1 (Δ = -0.263, "the foolishness of God is wiser than men," barely crossing from 0.251 to 0.514). Neither reveals anything like a hidden document.

#### Random Ablation

The obvious objection is that stripping tokens mutilates the dependency tree, and the resulting syntactic anomaly is what drives the tradition flip. Fair point. To test this, I run a random ablation control: for each scorable clause, randomly strip the same number of non-editorial, non-punctuation tokens (with identical cascade logic), repeated 100 times. The aggregate comparison using Set A:

| Metric     | Editorial     | Random (100 trials) |
| ---------- | ------------- | ------------------- |
| Flip rate  | 8.0% (24/299) | 7.6% ± 1.1%         |
| Mean ΔP(J) | +0.030        | +0.044              |

At the aggregate level, the flip rates are not statistically distinguishable (p=0.06). But the random baseline is not actually a clean test of "syntactic damage" because randomly removing *any* content-bearing token from a classified clause also removes grammatical structure that was contributing to the tradition classification. Every token carries tradition-associated morphosyntactic features. Removing a random verb or noun is never truly neutral. You are removing real tradition-carrying signal.

What matters is the pattern: random stripping produces *larger* mean deltas (+0.044 vs +0.030) while causing *fewer* threshold-crossing flips. Random deletion is noisy. It pushes scores around erratically. Editorial stripping is concentrated. It doesn't move the average clause much, but when it does move one, it moves it cleanly across the boundary. That is why per-flip robustness analysis is the real test, not the aggregate comparison. 21 of the J→G flips never occur under random deletion. Not once in 100 trials.

### Multi-Scale Analysis

I ran the analysis at five granularity levels (token windows, clauses, verses, 2-verse blocks, 3-verse blocks). The Gentile signal concentrates at smaller granularities. Mean P̄(J) across all books is 0.789 at the clause level vs. 0.875 at the 3-verse block level. Gentile-grammar material appears in short stretches, individual clauses and sentence fragments, that get diluted when you zoom out. This is exactly what you would expect if Jewish editorial framing wraps shorter Gentile-tradition content.

## 7. Discussion

80.7% hard accuracy from a grammar-only classifier does not, on first glance, look like much. But the AUC-ROC of 0.892 and the 96.3% accuracy on confident predictions tells a different story. The model has learned real tradition-discriminative features. The cases it gets "wrong" at the 0.5 threshold are genuinely ambiguous. When 10 models agree, they are almost always right. When they disagree, the text is probably sitting at a real intersection of traditions. Scoring it near 0.5 is the correct answer, not an error.

What the model picks up on is something like a grammatical accent. Patterns in case usage, verb mood distribution, participial chains, syntactic depth, coordination style. A Jewish-educated scribe constructs sentences differently than a Greek-educated one, even when writing about the same topic. That this holds after centuries of scribal transmission is itself remarkable.

The redaction results are where things get interesting. Three independent marker sets that share no lemmas produce 21 robust J→G flips. The reverse direction across the same three sets produces 2. A 10.5:1 asymmetry. The two G→J robust flips are marginal. The top J→G flips look like a different document underneath the editorial surface. Philippians 3:8: strip the Christological tokens and you get a gnosis formula, grammatically Gentile, centered on "the surpassing value of knowledge." I do not see how a 10.5:1 directional asymmetry with that kind of content specificity can be explained as syntactic damage.

Several of the robust flips resolve known exegetical problems. Hebrews 6 has been argued about for centuries because it seems to deny the possibility of post-apostasy repentance. Strip the sin/repentance vocabulary and the contradiction disappears. You get a coherent initiation text. The Philippians 2 kenosis hymn has the circular reference, "in the form of God... did not grasp at equality with God," which resolves when θεός is removed. 1 Peter 1 calls ancestral tradition "futile" (ματαίας ἀναστροφῆς πατροπαραδότου), a claim that reads more naturally as a standalone radical statement than as part of a Christian redemption narrative. In each case, the stripped text is not just grammatically different. It is more internally coherent than what we have in the canonical text.

I am not claiming these results prove any particular theory of NT composition. What I am claiming is that the grammatical evidence exists, it is robust across three independent marker sets, overwhelmingly unidirectional (21:2), and in several cases the stripped substrate resolves problems that have been debated for centuries.

### Limitations

1. **Binary threshold on a continuous signal.** The 0.5 decision boundary imposes a binary on what is fundamentally a continuum. Ancient Greek literary culture was not neatly partitioned. Authors like Philo and Josephus wrote within both traditions simultaneously, and educated scribes shared grammatical conventions that crossed ethnic and religious boundaries. The model's uncertainty in boundary cases is informative, not a flaw, but it does mean tradition classification should be understood as probabilistic.
2. **Editorial category selection.** The choice of which lemmas count as editorial markers comes from scholarly priors. The cross-set analysis partially addresses this concern: three independent marker sets all produce robust unidirectional flips, and the consistency across sets that share no lemmas suggests the signal is not an artifact of any particular category choice.
3. **CoNLL-U annotation quality.** Lower-confidence annotations in the NT CoNLL-U data may introduce noise.
4. **Training corpora.** Different training corpora would produce different results. The signal is relative to the grammatical profiles learned from these specific sources.
5. **Sentence length.** NT clauses average 14 tokens after splitting, shorter than the training data's 17 token median. One might suspect shorter clauses bias toward Gentile. They do not. The token distributions in the training data are nearly identical across traditions, with the Jewish mean at 22.5 tokens and the Gentile mean at 23.2. If anything, Gentile training sentences are slightly *longer*, which means the model scoring short NT clauses as Gentile cannot be explained by a length confound in the training data.

## 8. Conclusion

Morphosyntactic features alone carry enough signal to classify Greek literary tradition with high accuracy (AUC-ROC: 0.892, 96.3% on confident predictions), and that signal can detect editorial layers in the New Testament.

The redaction detection methodology, validated across three independent marker sets in both directions, identifies 21 robust J→G tradition flips and only 2 robust G→J flips. The original marker set produces 12 robust J→G flips (combined p < 8.87 × 10⁻²⁵), the Catholic set produces 3 (combined p < 9.71 × 10⁻⁷), and the expanded set produces 6 (combined p < 9.42 × 10⁻¹³). What remains after stripping reads coherently, scores Gentile, and in several cases (Hebrews 6, Philippians 2, Philippians 3:8, 1 Peter 1) resolves known exegetical difficulties in the canonical text.

All code, trained models, and analysis results are available at https://github.com/JaredMarshallTurcotte/morphosyntactic-fingerprinting

## References

- Assael, Y., Sommerschield, T., Shillingford, B., et al. (2022). Restoring and attributing ancient texts using deep neural networks. *Nature*, 603:280--283.
- Crane, G. R. (ed.). *Perseus Digital Library*. Tufts University. http://www.perseus.tufts.edu
- Eder, M., Piasecki, M., and Walkowiak, T. (2023). Morphosyntactic annotation in literary stylometry. *Information*, 15(4):211.
- Faigenbaum-Golovin, S., Shaus, A., et al. (2023). AI uncovers hidden authorial patterns in biblical texts. *Proceedings of the National Academy of Sciences*.
- Haug, D. T. T. and Jøhndal, M. L. (2008). Creating a parallel treebank of the old Indo-European Bible translations. *Proceedings of the Second Workshop on Language Technology for Cultural Heritage Data (LaTeCH 2008)*, 27--34.
- Kenny, A. (1986). *A Stylometric Study of the New Testament*. Oxford University Press.
- Morton, A. Q. (1978). *Literary Detection: How to Prove Authorship and Fraud in Literature and Documents*. Scribner's.
- Sommerschield, T., Assael, Y., Pavlopoulos, J., et al. (2023). Machine learning for ancient languages: A survey. *Computational Linguistics*, 49(3):703--747.
- Tauber, J. K. (2017). *MorphGNT: Morphologically tagged Greek New Testament*. https://github.com/morphgnt/sblgnt
