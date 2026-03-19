"""
Hierarchical feature extractor for HMSM Legal Move Mapper.

Extracts features at four hierarchical levels:
  Level 1: Semantic constraints (lemma, Aktionsart, argument structure)
  Level 2: Syntactic constraints (clause type, word order, subject/object)
  Level 3: Discourse constraints (topic continuity, information structure)
  Level 4: Micro-context (local morphology, clause markers, particles)
"""

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

from .conllu_parser import ConlluToken, Sentence
from .morphology_normalizer import MorphologyNormalizer


@dataclass
class SemanticFeatures:
    """Level 1: Semantic constraint features."""

    lemma: str = ""
    aktionsart: str = "unknown"      # telic, atelic, punctual, durative
    argument_structure: str = "unknown"  # transitive, intransitive, ditransitive
    is_stative: bool = False
    semantic_class: str = "unknown"  # motion, cognition, speech, etc.

    # Numeric encodings for neural network
    aktionsart_idx: int = 0
    arg_structure_idx: int = 0
    stative_idx: int = 0
    semantic_class_idx: int = 0


@dataclass
class SyntacticFeatures:
    """Level 2: Syntactic constraint features."""

    clause_type: str = "main"        # main, subordinate, participial, infinitival, relative
    word_order_position: int = 0     # Position in clause (0-indexed)
    clause_length: int = 0           # Total words in clause
    relative_position: float = 0.0   # Position / length
    deprel: str = ""                 # Dependency relation
    head_pos: str = ""               # POS of head
    subject_type: str = "unknown"    # nominal, pronominal, null
    object_type: str = "unknown"     # nominal, pronominal, clausal, none

    # Numeric encodings
    clause_type_idx: int = 0
    deprel_idx: int = 0
    head_pos_idx: int = 0
    subject_type_idx: int = 0
    object_type_idx: int = 0

    # New features
    tree_depth: int = 0              # Distance to root in dependency tree
    genitive_chain_depth: int = 0    # Depth of genitive nmod chain


@dataclass
class DiscourseFeatures:
    """Level 3: Discourse constraint features."""

    topic_continuity: float = 0.0    # 0=new, 1=continued
    information_status: str = "new"  # given, new, accessible
    rhetorical_mode: str = "narrative"  # narrative, argument, exhortation, exposition
    sentence_position: int = 0       # Position in paragraph/text
    is_topic: bool = False           # Is this the topic of the sentence
    is_focus: bool = False           # Is this the focus

    # Numeric encodings
    information_status_idx: int = 0
    rhetorical_mode_idx: int = 0

    # New features
    kai_rate: float = 0.0            # Fraction of CCONJ+cc tokens whose lemma is καί


@dataclass
class MicroContextFeatures:
    """Level 4: Micro-context features."""

    # Previous tokens morphology (up to 3)
    prev_forms: List[str] = field(default_factory=list)
    prev_lemmas: List[str] = field(default_factory=list)
    prev_pos: List[str] = field(default_factory=list)
    prev_morph_codes: List[str] = field(default_factory=list)

    # Next tokens morphology (up to 2)
    next_forms: List[str] = field(default_factory=list)
    next_lemmas: List[str] = field(default_factory=list)
    next_pos: List[str] = field(default_factory=list)
    next_morph_codes: List[str] = field(default_factory=list)

    # Clause markers present in context
    clause_markers: List[str] = field(default_factory=list)  # καί, δέ, γάρ, οὖν, etc.

    # Particles present in context
    particles: List[str] = field(default_factory=list)  # μέν, ἄν, δή, γε, etc.

    # Position flags
    is_clause_initial: bool = False
    is_clause_final: bool = False
    follows_particle: bool = False
    precedes_particle: bool = False

    # Numeric vectors for neural network
    prev_morph_vectors: Optional[np.ndarray] = None  # [3, morph_dim]
    next_morph_vectors: Optional[np.ndarray] = None  # [2, morph_dim]
    clause_marker_vector: Optional[np.ndarray] = None  # [num_markers]
    particle_vector: Optional[np.ndarray] = None       # [num_particles]

    # New features
    article_pattern: Optional[np.ndarray] = None       # [3] binary: prev/curr/next article


class MentionDescriptor(NamedTuple):
    """Lightweight referent descriptor for cross-sentence tracking."""
    lemma: str
    upos: str       # NOUN, PROPN, PRON
    deprel: str     # nsubj, obj, iobj, obl, nsubj:pass
    person: str     # 1, 2, 3, or _
    number: str     # Sing, Plur, Dual, or _
    gender: str     # Masc, Fem, Neut, or _
    is_definite: bool


@dataclass
class HierarchicalFeatures:
    """Combined features from all four levels."""

    token_position: int
    token: ConlluToken

    semantic: SemanticFeatures = field(default_factory=SemanticFeatures)
    syntactic: SyntacticFeatures = field(default_factory=SyntacticFeatures)
    discourse: DiscourseFeatures = field(default_factory=DiscourseFeatures)
    context: MicroContextFeatures = field(default_factory=MicroContextFeatures)


# POS tags used for n-gram features (collapsed to keep dimension manageable)
_POS_NGRAM_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ', 'PART', 'OTHER']
_POS_NGRAM_MAP = {
    'NOUN': 'NOUN', 'PROPN': 'NOUN', 'VERB': 'VERB', 'AUX': 'VERB',
    'ADJ': 'ADJ', 'ADV': 'ADV', 'PRON': 'PRON', 'DET': 'DET',
    'ADP': 'ADP', 'CCONJ': 'CONJ', 'SCONJ': 'CONJ', 'PART': 'PART',
}

# Case tags for transition bigrams
_CASE_TAGS = ['Nom', 'Gen', 'Dat', 'Acc', 'Voc']

# Pre-compute n-gram vocabularies
_POS_BIGRAM_VOCAB = [(a, b) for a in _POS_NGRAM_TAGS for b in _POS_NGRAM_TAGS]  # 100
_POS_BIGRAM_IDX = {bg: i for i, bg in enumerate(_POS_BIGRAM_VOCAB)}

# Top POS trigrams (most linguistically informative, not all 1000)
_POS_TRIGRAM_VOCAB = [
    # Article-noun patterns
    ('DET', 'NOUN', 'NOUN'), ('DET', 'ADJ', 'NOUN'), ('DET', 'NOUN', 'ADP'),
    ('ADP', 'DET', 'NOUN'), ('NOUN', 'DET', 'NOUN'),
    # Verb patterns
    ('CONJ', 'VERB', 'NOUN'), ('CONJ', 'VERB', 'DET'), ('VERB', 'DET', 'NOUN'),
    ('NOUN', 'VERB', 'NOUN'), ('VERB', 'ADP', 'DET'), ('VERB', 'NOUN', 'NOUN'),
    ('PRON', 'VERB', 'NOUN'), ('VERB', 'ADV', 'VERB'),
    # Conjunction patterns
    ('CONJ', 'DET', 'NOUN'), ('CONJ', 'NOUN', 'VERB'), ('CONJ', 'PRON', 'VERB'),
    # Particle patterns
    ('PART', 'VERB', 'NOUN'), ('NOUN', 'PART', 'VERB'),
    # Preposition patterns
    ('ADP', 'NOUN', 'VERB'), ('ADP', 'DET', 'ADJ'), ('ADP', 'PRON', 'VERB'),
    # Adjective patterns
    ('ADJ', 'NOUN', 'VERB'), ('DET', 'ADJ', 'ADJ'), ('NOUN', 'ADJ', 'NOUN'),
    # Adverb patterns
    ('ADV', 'VERB', 'NOUN'), ('ADV', 'ADV', 'VERB'),
]  # 26
_POS_TRIGRAM_IDX = {tg: i for i, tg in enumerate(_POS_TRIGRAM_VOCAB)}

_CASE_BIGRAM_VOCAB = [(a, b) for a in _CASE_TAGS for b in _CASE_TAGS]  # 25
_CASE_BIGRAM_IDX = {bg: i for i, bg in enumerate(_CASE_BIGRAM_VOCAB)}

# Mood transition bigrams
_MOOD_TAGS = ['Ind', 'Sub', 'Imp', 'Inf', 'Part', 'Opt']
_MOOD_BIGRAM_VOCAB = [(a, b) for a in _MOOD_TAGS for b in _MOOD_TAGS]  # 36
_MOOD_BIGRAM_IDX = {bg: i for i, bg in enumerate(_MOOD_BIGRAM_VOCAB)}

# Tense transition bigrams
_TENSE_TAGS = ['Past', 'Pres', 'Fut']
_TENSE_BIGRAM_VOCAB = [(a, b) for a in _TENSE_TAGS for b in _TENSE_TAGS]  # 9
_TENSE_BIGRAM_IDX = {bg: i for i, bg in enumerate(_TENSE_BIGRAM_VOCAB)}

# Voice transition bigrams
_VOICE_TAGS = ['Act', 'Mid', 'Pass']
_VOICE_BIGRAM_VOCAB = [(a, b) for a in _VOICE_TAGS for b in _VOICE_TAGS]  # 9
_VOICE_BIGRAM_IDX = {bg: i for i, bg in enumerate(_VOICE_BIGRAM_VOCAB)}

# Clause-linking marker bigrams — sequence of connectives (not all 17x17, use top markers)
_MARKER_TAGS = ['καί', 'δέ', 'γάρ', 'οὖν', 'ἀλλά', 'μέν', 'ὅτι', 'ἵνα', 'ὡς', 'OTHER_MARKER']
_MARKER_BIGRAM_VOCAB = [(a, b) for a in _MARKER_TAGS for b in _MARKER_TAGS]  # 100
_MARKER_BIGRAM_IDX = {bg: i for i, bg in enumerate(_MARKER_BIGRAM_VOCAB)}
_ALL_MARKERS_SET = {'καί', 'δέ', 'γάρ', 'οὖν', 'ἀλλά', 'μέν', 'τε', 'ὅτι', 'ἵνα',
                    'εἰ', 'ἐάν', 'ὡς', 'ὅτε', 'ἐπεί', 'διότι', 'ὥστε', 'πρίν'}

# Stylometric features dimension
_STYLOMETRIC_DIM = 10  # asyndeton, articular infinitive, genitive absolute, preposition diversity, etc.

# Additional syntactic profile features
_SYNTACTIC_PROFILE_DIM = 18  # verb position, tree depth, participial chains, Wackernagel, etc.

# Total n-gram dimensions: 100 + 26 + 25 + 36 + 9 + 9 + 100 = 305
_NGRAM_DIM = (len(_POS_BIGRAM_VOCAB) + len(_POS_TRIGRAM_VOCAB) + len(_CASE_BIGRAM_VOCAB)
              + len(_MOOD_BIGRAM_VOCAB) + len(_TENSE_BIGRAM_VOCAB) + len(_VOICE_BIGRAM_VOCAB)
              + len(_MARKER_BIGRAM_VOCAB))

SENTENCE_GRAMMAR_PROFILE_DIM = 58 + _NGRAM_DIM + _STYLOMETRIC_DIM + _SYNTACTIC_PROFILE_DIM  # 58 + 305 + 10 + 18 = 391


def _extract_ngram_features(tokens) -> np.ndarray:
    """Extract POS, case, mood, tense, and voice n-gram features."""
    n = max(len(tokens) - 1, 1)

    # Map POS tags
    pos_seq = [_POS_NGRAM_MAP.get(t.upos, 'OTHER') for t in tokens]

    # POS bigrams (100d, normalized by sentence length)
    pos_bigrams = np.zeros(len(_POS_BIGRAM_VOCAB), dtype=np.float32)
    for i in range(len(pos_seq) - 1):
        bg = (pos_seq[i], pos_seq[i + 1])
        idx = _POS_BIGRAM_IDX.get(bg)
        if idx is not None:
            pos_bigrams[idx] += 1
    if n > 0:
        pos_bigrams /= n

    # POS trigrams (26d, normalized)
    pos_trigrams = np.zeros(len(_POS_TRIGRAM_VOCAB), dtype=np.float32)
    n_tri = max(len(pos_seq) - 2, 1)
    for i in range(len(pos_seq) - 2):
        tg = (pos_seq[i], pos_seq[i + 1], pos_seq[i + 2])
        idx = _POS_TRIGRAM_IDX.get(tg)
        if idx is not None:
            pos_trigrams[idx] += 1
    if n_tri > 0:
        pos_trigrams /= n_tri

    # Case transition bigrams (25d, normalized)
    case_bigrams = np.zeros(len(_CASE_BIGRAM_VOCAB), dtype=np.float32)
    case_seq = [t.get_feat('Case', '_') for t in tokens]
    n_case_pairs = 0
    for i in range(len(case_seq) - 1):
        if case_seq[i] != '_' and case_seq[i + 1] != '_':
            bg = (case_seq[i], case_seq[i + 1])
            idx = _CASE_BIGRAM_IDX.get(bg)
            if idx is not None:
                case_bigrams[idx] += 1
            n_case_pairs += 1
    if n_case_pairs > 0:
        case_bigrams /= n_case_pairs

    # Mood transition bigrams (36d) — between adjacent verbs/participles
    mood_bigrams = np.zeros(len(_MOOD_BIGRAM_VOCAB), dtype=np.float32)
    mood_seq = []
    for t in tokens:
        m = t.get_feat('Mood', '_')
        if m != '_':
            mood_seq.append(m)
    n_mood_pairs = 0
    for i in range(len(mood_seq) - 1):
        bg = (mood_seq[i], mood_seq[i + 1])
        idx = _MOOD_BIGRAM_IDX.get(bg)
        if idx is not None:
            mood_bigrams[idx] += 1
        n_mood_pairs += 1
    if n_mood_pairs > 0:
        mood_bigrams /= n_mood_pairs

    # Tense transition bigrams (9d) — between adjacent verbs
    tense_bigrams = np.zeros(len(_TENSE_BIGRAM_VOCAB), dtype=np.float32)
    tense_seq = []
    for t in tokens:
        te = t.get_feat('Tense', '_')
        if te != '_':
            tense_seq.append(te)
    n_tense_pairs = 0
    for i in range(len(tense_seq) - 1):
        bg = (tense_seq[i], tense_seq[i + 1])
        idx = _TENSE_BIGRAM_IDX.get(bg)
        if idx is not None:
            tense_bigrams[idx] += 1
        n_tense_pairs += 1
    if n_tense_pairs > 0:
        tense_bigrams /= n_tense_pairs

    # Voice transition bigrams (9d) — between adjacent verbs
    voice_bigrams = np.zeros(len(_VOICE_BIGRAM_VOCAB), dtype=np.float32)
    voice_seq = []
    for t in tokens:
        vo = t.get_feat('Voice', '_')
        if vo != '_':
            voice_seq.append(vo)
    n_voice_pairs = 0
    for i in range(len(voice_seq) - 1):
        bg = (voice_seq[i], voice_seq[i + 1])
        idx = _VOICE_BIGRAM_IDX.get(bg)
        if idx is not None:
            voice_bigrams[idx] += 1
        n_voice_pairs += 1
    if n_voice_pairs > 0:
        voice_bigrams /= n_voice_pairs

    # Clause-linking marker bigrams (100d) — sequence of connectives through sentence
    marker_bigrams = np.zeros(len(_MARKER_BIGRAM_VOCAB), dtype=np.float32)
    marker_seq = []
    for t in tokens:
        if t.lemma in _ALL_MARKERS_SET or t.upos in ('CCONJ', 'SCONJ'):
            tag = t.lemma if t.lemma in dict.fromkeys(_MARKER_TAGS[:-1]) else 'OTHER_MARKER'
            marker_seq.append(tag)
    n_marker_pairs = 0
    for i in range(len(marker_seq) - 1):
        bg = (marker_seq[i], marker_seq[i + 1])
        idx = _MARKER_BIGRAM_IDX.get(bg)
        if idx is not None:
            marker_bigrams[idx] += 1
        n_marker_pairs += 1
    if n_marker_pairs > 0:
        marker_bigrams /= n_marker_pairs

    return np.concatenate([pos_bigrams, pos_trigrams, case_bigrams,
                           mood_bigrams, tense_bigrams, voice_bigrams,
                           marker_bigrams])


def _extract_stylometric_features(tokens) -> np.ndarray:
    """
    Extract classical Greek stylometric features.
    These are specifically cited in scholarship on Semitic vs Atticist Greek.
    """
    n = max(len(tokens), 1)

    # ── 1. Asyndeton rate ──
    # A clause/sentence beginning without a connecting conjunction or particle.
    # Check if first word is NOT a conjunction, particle, or connective.
    CONNECTIVES = {'καί', 'δέ', 'γάρ', 'οὖν', 'ἀλλά', 'μέν', 'τε', 'ὅτι', 'ἵνα',
                   'εἰ', 'ἐάν', 'ὡς', 'ὅτε', 'ἐπεί', 'διότι', 'ὥστε', 'πρίν',
                   'μέντοι', 'καίτοι', 'ἆρα', 'τοίνυν', 'τοιγαροῦν'}
    is_asyndeton = 0.0
    if tokens and tokens[0].lemma not in CONNECTIVES and tokens[0].upos not in ('CCONJ', 'SCONJ'):
        is_asyndeton = 1.0

    # ── 2. Articular infinitive ──
    # τό/τοῦ/τῷ + infinitive — distinctly literary Greek, rare in Semitic-influenced writing.
    articular_inf_count = 0
    for i in range(len(tokens) - 1):
        if (tokens[i].upos == 'DET' and tokens[i].lemma == 'ὁ'
                and tokens[i + 1].get_feat('Mood', '_') == 'Inf'):
            articular_inf_count += 1
    # Also check non-adjacent (article ... infinitive within 3 tokens)
    for i in range(len(tokens) - 2):
        if (tokens[i].upos == 'DET' and tokens[i].lemma == 'ὁ'):
            for j in range(i + 2, min(i + 4, len(tokens))):
                if tokens[j].get_feat('Mood', '_') == 'Inf':
                    articular_inf_count += 1
                    break

    # ── 3. Genitive absolute ──
    # Genitive participle whose subject is in genitive and not shared with main clause.
    # Approximation: genitive participle token (Mood=Part, Case=Gen)
    gen_absolute_count = 0
    for t in tokens:
        if (t.get_feat('Mood', '_') == 'Part'
                and t.get_feat('Case', '_') == 'Gen'
                and t.upos in ('VERB', 'AUX')):
            gen_absolute_count += 1

    # ── 4. Preposition diversity ──
    preps = [t.lemma for t in tokens if t.upos == 'ADP']
    n_preps = len(preps)
    unique_preps = len(set(preps))
    # Common Semitic-influenced prepositions
    SEMITIC_PREPS = {'ἐν', 'εἰς', 'ἀπό', 'ἐκ', 'πρός'}
    semitic_prep_count = sum(1 for p in preps if p in SEMITIC_PREPS)

    # ── 5. Periphrastic construction ──
    # εἰμί + participle — more common in later/Semitic-influenced Greek
    periphrastic_count = 0
    for i in range(len(tokens) - 1):
        if (tokens[i].lemma == 'εἰμί' and tokens[i + 1].get_feat('Mood', '_') == 'Part'):
            periphrastic_count += 1
        elif (tokens[i].get_feat('Mood', '_') == 'Part' and
              i + 1 < len(tokens) and tokens[i + 1].lemma == 'εἰμί'):
            periphrastic_count += 1

    return np.array([
        is_asyndeton,                                          # asyndeton flag
        articular_inf_count / n,                               # articular infinitive density
        gen_absolute_count / n,                                # genitive absolute density
        unique_preps / max(n_preps, 1),                        # preposition diversity
        n_preps / n,                                           # preposition density
        semitic_prep_count / max(n_preps, 1),                  # Semitic preposition ratio
        periphrastic_count / n,                                # periphrastic construction density
        # Article usage patterns
        sum(1 for t in tokens if t.upos == 'DET') / n,        # article density
        # Pronoun density (Semitic Greek tends toward more explicit pronouns)
        sum(1 for t in tokens if t.upos == 'PRON') / n,       # pronoun density
        # Mean word "rank" in clause (approximation of word order freedom)
        np.std([t.int_id for t in tokens if t.int_id > 0]) / max(n, 1) if any(t.int_id > 0 for t in tokens) else 0.0,
    ], dtype=np.float32)


def _extract_syntactic_profile(sentence) -> np.ndarray:
    """
    Extract deeper syntactic structure features.
    Verb position, tree depth, participial chains, Wackernagel, relative clauses, negation.
    """
    tokens = sentence.word_tokens
    n = max(len(tokens), 1)

    # ── 1. Verb position in clause (3d) ──
    # Where does the main verb fall? Approximate using root verb position.
    verb_initial = 0.0
    verb_medial = 0.0
    verb_final = 0.0
    root_verbs = [t for t in tokens if t.deprel == 'root' and t.upos in ('VERB', 'AUX')]
    if root_verbs and len(tokens) >= 3:
        root_pos = tokens.index(root_verbs[0]) if root_verbs[0] in tokens else -1
        if root_pos >= 0:
            relative = root_pos / max(len(tokens) - 1, 1)
            if relative < 0.33:
                verb_initial = 1.0
            elif relative < 0.67:
                verb_medial = 1.0
            else:
                verb_final = 1.0

    # ── 2. Dependency tree depth (3d) ──
    # Build parent map and compute depths
    id_to_token = {}
    for t in tokens:
        if t.int_id > 0:
            id_to_token[str(t.int_id)] = t

    def _get_depth(tok, visited=None):
        if visited is None:
            visited = set()
        if tok.head == '0' or tok.head in visited:
            return 0
        visited.add(tok.head)
        parent = id_to_token.get(tok.head)
        if parent is None:
            return 0
        return 1 + _get_depth(parent, visited)

    depths = [_get_depth(t) for t in tokens if t.int_id > 0]
    max_depth = max(depths) if depths else 0
    mean_depth = np.mean(depths) if depths else 0.0
    depth_std = np.std(depths) if depths else 0.0

    # ── 3. Participial chain length (2d) ──
    # Count consecutive participles before main verb
    max_part_chain = 0
    current_chain = 0
    for t in tokens:
        if t.get_feat('Mood', '_') == 'Part':
            current_chain += 1
            max_part_chain = max(max_part_chain, current_chain)
        else:
            current_chain = 0
    total_participles = sum(1 for t in tokens if t.get_feat('Mood', '_') == 'Part')

    # ── 4. Wackernagel position adherence (2d) ──
    # In classical Greek, clitics/particles tend to go in second position in their clause.
    # Check how often particles appear in position 2 vs elsewhere.
    CLITIC_PARTICLES = {'μέν', 'δέ', 'γάρ', 'τε', 'ἄν', 'γε', 'δή', 'που', 'τοι', 'περ'}
    particles_in_second = 0
    particles_elsewhere = 0
    for i, t in enumerate(tokens):
        if t.lemma in CLITIC_PARTICLES:
            if i == 1:  # second position (0-indexed)
                particles_in_second += 1
            else:
                particles_elsewhere += 1
    total_clitics = particles_in_second + particles_elsewhere
    wackernagel_rate = particles_in_second / max(total_clitics, 1)

    # ── 5. Relative clause features (2d) ──
    # Count relative pronouns and relative clause deprels
    rel_pronouns = sum(1 for t in tokens if t.get_feat('PronType', '_') == 'Rel')
    rel_clauses = sum(1 for t in tokens if t.deprel in ('acl:relcl', 'acl'))

    # ── 6. Negation patterns (3d) ──
    # οὐ for indicative/factual, μή for subjunctive/imperative/infinitive
    ou_count = sum(1 for t in tokens if t.lemma in ('οὐ', 'οὐκ', 'οὐχ', 'οὐδέ', 'οὔτε', 'οὐδείς'))
    me_count = sum(1 for t in tokens if t.lemma in ('μή', 'μηδέ', 'μήτε', 'μηδείς'))
    total_neg = ou_count + me_count

    # ── 7. Hapax POS bigram rate (1d) ──
    # How many POS bigrams appear only once — measures syntactic unusualness
    from collections import Counter
    pos_bg_counts = Counter()
    for i in range(len(tokens) - 1):
        pos_bg_counts[(tokens[i].upos, tokens[i + 1].upos)] += 1
    hapax_bigrams = sum(1 for c in pos_bg_counts.values() if c == 1)
    total_bigrams = max(len(tokens) - 1, 1)

    # ── 8. Subject-verb distance (2d) ──
    # How far is the subject from its verb? Large distance = more complex word order.
    sv_distances = []
    for t in tokens:
        if t.deprel in ('nsubj', 'nsubj:pass'):
            head = id_to_token.get(t.head)
            if head and head.upos in ('VERB', 'AUX'):
                sv_distances.append(abs(t.int_id - head.int_id))
    mean_sv_dist = np.mean(sv_distances) if sv_distances else 0.0
    max_sv_dist = max(sv_distances) if sv_distances else 0

    return np.array([
        # Verb position (3d)
        verb_initial,
        verb_medial,
        verb_final,
        # Tree depth (3d)
        max_depth / max(n, 1),                      # normalized max depth
        mean_depth / max(n, 1),                      # normalized mean depth
        depth_std / max(n, 1),                       # depth variability
        # Participial chains (2d)
        max_part_chain / max(n, 1),                  # longest participle chain
        total_participles / n,                       # participle density
        # Wackernagel (2d)
        wackernagel_rate,                            # clitic second-position rate
        total_clitics / n,                           # clitic density
        # Relative clauses (2d)
        rel_pronouns / n,                            # relative pronoun density
        rel_clauses / n,                             # relative clause density
        # Negation (3d)
        ou_count / n,                                # οὐ density
        me_count / n,                                # μή density
        ou_count / max(total_neg, 1),                # οὐ vs μή ratio
        # Hapax bigrams (1d)
        hapax_bigrams / total_bigrams,               # POS bigram uniqueness
        # Subject-verb distance (2d)
        mean_sv_dist / max(n, 1),                    # mean subject-verb distance
        max_sv_dist / max(n, 1),                     # max subject-verb distance
    ], dtype=np.float32)


def extract_sentence_grammar_profile(sentence: 'Sentence') -> np.ndarray:
    """
    Extract sentence-level grammar profile features.

    These capture aggregate grammatical patterns across an entire sentence,
    not visible from a single token's local context. All features are
    BERT-independent — derived from POS, morphology, deprel, and lemma only.

    Returns:
        np.ndarray of shape [SENTENCE_GRAMMAR_PROFILE_DIM]
    """
    tokens = sentence.word_tokens
    n = len(tokens) if tokens else 1  # avoid div by zero

    # ── 1. Subordination ratio (5d) ──
    # Counts of coordinating vs subordinating markers
    COORD_MARKERS = {'καί', 'δέ', 'ἀλλά', 'τε', 'μέν'}
    SUBORD_MARKERS = {'ὅτι', 'ἵνα', 'ἐάν', 'εἰ', 'ὡς', 'ὅτε', 'ἐπεί', 'διότι', 'ὥστε', 'πρίν'}
    coord_count = sum(1 for t in tokens if t.lemma in COORD_MARKERS)
    subord_count = sum(1 for t in tokens if t.lemma in SUBORD_MARKERS)
    total_markers = coord_count + subord_count
    subordination_features = np.array([
        coord_count / n,                                    # coord density
        subord_count / n,                                   # subord density
        subord_count / max(total_markers, 1),               # subord ratio
        1.0 if subord_count > coord_count else 0.0,         # hypotactic flag
        total_markers / n,                                  # overall marker density
    ], dtype=np.float32)

    # ── 2. Particle density and profile (6d) ──
    PARTICLES = {'μέν', 'ἄν', 'δή', 'γε', 'που', 'τοι', 'περ', 'μήν', 'ἆρα', 'καίτοι', 'μέντοι'}
    particle_count = sum(1 for t in tokens if t.lemma in PARTICLES)
    has_men_de = any(t.lemma == 'μέν' for t in tokens) and any(t.lemma == 'δέ' for t in tokens)
    has_an = any(t.lemma == 'ἄν' for t in tokens)
    has_ge = any(t.lemma == 'γε' for t in tokens)
    particle_features = np.array([
        particle_count / n,                # particle density
        1.0 if has_men_de else 0.0,        # μέν...δέ correlation (Atticist marker)
        1.0 if has_an else 0.0,            # ἄν present (conditional/potential)
        1.0 if has_ge else 0.0,            # γε present (emphatic)
        min(particle_count, 5) / 5.0,      # particle richness (capped)
        particle_count / max(coord_count + subord_count, 1),  # particle-to-marker ratio
    ], dtype=np.float32)

    # ── 3. Verb chain patterns (12d) ──
    # Tense/mood distribution across verbs in sentence
    verbs = [t for t in tokens if t.upos in ('VERB', 'AUX')]
    n_verbs = max(len(verbs), 1)
    tense_counts = {'Past': 0, 'Pres': 0, 'Fut': 0}
    mood_counts = {'Ind': 0, 'Sub': 0, 'Imp': 0, 'Inf': 0, 'Part': 0, 'Opt': 0}
    voice_counts = {'Act': 0, 'Mid': 0, 'Pass': 0}
    for v in verbs:
        t = v.get_feat('Tense', '_')
        m = v.get_feat('Mood', '_')
        vo = v.get_feat('Voice', '_')
        if t in tense_counts:
            tense_counts[t] += 1
        if m in mood_counts:
            mood_counts[m] += 1
        if vo in voice_counts:
            voice_counts[vo] += 1
    verb_chain_features = np.array([
        len(verbs) / n,                          # verb density
        tense_counts['Past'] / n_verbs,          # aorist/imperfect ratio
        tense_counts['Pres'] / n_verbs,          # present ratio
        tense_counts['Fut'] / n_verbs,           # future ratio
        mood_counts['Ind'] / n_verbs,            # indicative ratio
        mood_counts['Sub'] / n_verbs,            # subjunctive ratio
        mood_counts['Part'] / n_verbs,           # participle ratio
        mood_counts['Inf'] / n_verbs,            # infinitive ratio
        mood_counts['Imp'] / n_verbs,            # imperative ratio
        voice_counts['Act'] / n_verbs,           # active ratio
        voice_counts['Mid'] / n_verbs,           # middle ratio
        voice_counts['Pass'] / n_verbs,          # passive ratio
    ], dtype=np.float32)

    # ── 4. Case distribution (6d) ──
    case_counts = {'Nom': 0, 'Gen': 0, 'Dat': 0, 'Acc': 0, 'Voc': 0}
    n_cased = 0
    for t in tokens:
        c = t.get_feat('Case', '_')
        if c in case_counts:
            case_counts[c] += 1
            n_cased += 1
    n_cased = max(n_cased, 1)
    case_features = np.array([
        case_counts['Nom'] / n_cased,
        case_counts['Gen'] / n_cased,
        case_counts['Dat'] / n_cased,
        case_counts['Acc'] / n_cased,
        case_counts['Voc'] / n_cased,
        case_counts['Gen'] / max(case_counts['Nom'], 1),  # gen-to-nom ratio (genitive chain indicator)
    ], dtype=np.float32)

    # ── 5. POS distribution (10d) ──
    pos_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0, 'PRON': 0,
                  'DET': 0, 'ADP': 0, 'CCONJ': 0, 'SCONJ': 0, 'PART': 0}
    for t in tokens:
        if t.upos in pos_counts:
            pos_counts[t.upos] += 1
    pos_features = np.array([
        pos_counts[p] / n for p in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON',
                                     'DET', 'ADP', 'CCONJ', 'SCONJ', 'PART']
    ], dtype=np.float32)

    # ── 6. Deprel distribution (8d) ──
    deprel_groups = {
        'core': {'nsubj', 'obj', 'iobj', 'nsubj:pass'},
        'oblique': {'obl', 'obl:agent'},
        'modifier': {'amod', 'advmod', 'nmod'},
        'clause': {'advcl', 'ccomp', 'xcomp', 'csubj', 'acl', 'acl:relcl'},
        'coord': {'conj', 'cc'},
        'determiner': {'det'},
        'case_mark': {'case', 'mark'},
        'other': set(),
    }
    deprel_counts = {k: 0 for k in deprel_groups}
    for t in tokens:
        found = False
        for group, rels in deprel_groups.items():
            if group == 'other':
                continue
            if t.deprel in rels:
                deprel_counts[group] += 1
                found = True
                break
        if not found:
            deprel_counts['other'] += 1
    deprel_features = np.array([
        deprel_counts[g] / n for g in ['core', 'oblique', 'modifier', 'clause',
                                        'coord', 'determiner', 'case_mark', 'other']
    ], dtype=np.float32)

    # ── 7. Morphological transition bigrams (7d) ──
    # Count transitions between morphological categories in adjacent tokens
    case_transitions = 0  # number of case changes between adjacent nominals
    mood_transitions = 0  # number of mood changes between adjacent verbs
    n_nominal_pairs = 0
    n_verb_pairs = 0
    for i in range(len(tokens) - 1):
        t1, t2 = tokens[i], tokens[i + 1]
        c1, c2 = t1.get_feat('Case', '_'), t2.get_feat('Case', '_')
        if c1 != '_' and c2 != '_':
            n_nominal_pairs += 1
            if c1 != c2:
                case_transitions += 1
        m1, m2 = t1.get_feat('Mood', '_'), t2.get_feat('Mood', '_')
        if m1 != '_' and m2 != '_':
            n_verb_pairs += 1
            if m1 != m2:
                mood_transitions += 1
    # Article-noun adjacency
    article_noun_pairs = sum(
        1 for i in range(len(tokens) - 1)
        if tokens[i].upos == 'DET' and tokens[i + 1].upos in ('NOUN', 'ADJ', 'PROPN')
    )
    bigram_features = np.array([
        case_transitions / max(n_nominal_pairs, 1),     # case variability
        mood_transitions / max(n_verb_pairs, 1),        # mood variability
        article_noun_pairs / n,                         # article-noun density
        n_nominal_pairs / n,                            # nominal pair density
        n_verb_pairs / n,                               # verb pair density
        # POS bigram entropy approximation: unique POS bigrams / total
        len(set((tokens[i].upos, tokens[i+1].upos) for i in range(len(tokens)-1))) / max(len(tokens)-1, 1),
        # Sentence length (log-scaled)
        np.log1p(n),
    ], dtype=np.float32)

    # ── 8. Clause structure (4d) ──
    # Use deprel to approximate clause boundaries
    n_clausal = sum(1 for t in tokens if t.deprel in
                    {'advcl', 'ccomp', 'xcomp', 'csubj', 'acl', 'acl:relcl', 'parataxis'})
    n_root = sum(1 for t in tokens if t.deprel == 'root')
    clause_features = np.array([
        n_clausal / n,                     # clausal dependency density
        n_root / n,                        # root density (usually 1/n)
        n_clausal / max(n_root, 1),        # clauses per root
        (n_clausal + subord_count) / n,    # overall subordination density
    ], dtype=np.float32)

    # ── 9. N-gram features (305d) ──
    ngram_features = _extract_ngram_features(tokens)

    # ── 10. Stylometric features (10d) ──
    stylometric_features = _extract_stylometric_features(tokens)

    # ── 11. Syntactic profile (18d) ──
    syntactic_profile = _extract_syntactic_profile(sentence)

    # Concatenate all
    profile = np.concatenate([
        subordination_features,   # 5d
        particle_features,        # 6d
        verb_chain_features,      # 12d
        case_features,            # 6d
        pos_features,             # 10d
        deprel_features,          # 8d
        bigram_features,          # 7d
        clause_features,          # 4d
        ngram_features,           # 305d (100 POS bg + 26 POS tg + 25 case + 36 mood + 9 tense + 9 voice + 100 marker)
        stylometric_features,     # 10d (asyndeton, articular inf, gen absolute, prep diversity, etc.)
        syntactic_profile,        # 18d (verb position, tree depth, participial chains, Wackernagel, etc.)
    ])  # total: 391d

    assert profile.shape[0] == SENTENCE_GRAMMAR_PROFILE_DIM, \
        f"Expected {SENTENCE_GRAMMAR_PROFILE_DIM}, got {profile.shape[0]}"

    return profile


class HierarchicalFeatureExtractor:
    """
    Extracts hierarchical features from Greek text for HMSM.
    """

    # Common Greek clause markers
    CLAUSE_MARKERS = [
        'καί', 'δέ', 'γάρ', 'οὖν', 'ἀλλά', 'μέν', 'τε', 'ὅτι', 'ἵνα',
        'εἰ', 'ἐάν', 'ὡς', 'ὅτε', 'ἐπεί', 'διότι', 'ὥστε', 'πρίν',
    ]

    # Common Greek particles
    PARTICLES = [
        'μέν', 'ἄν', 'δή', 'γε', 'που', 'τοι', 'περ', 'νύν', 'δῆτα',
        'μήν', 'οὖν', 'ἆρα', 'καίτοι', 'μέντοι', 'ἤδη',
    ]

    # Aktionsart indicators by lemma patterns
    AKTIONSART_PATTERNS = {
        'punctual': ['θνῄσκω', 'ἀποθνῄσκω', 'πίπτω', 'ἵστημι', 'εὑρίσκω'],
        'telic': ['ποιέω', 'γράφω', 'κτίζω', 'οἰκοδομέω', 'πληρόω'],
        'atelic': ['βαδίζω', 'περιπατέω', 'ζάω', 'ἀγαπάω', 'πιστεύω'],
        'durative': ['μένω', 'οἰκέω', 'βασιλεύω', 'ἄρχω', 'κρατέω'],
    }

    # Verb semantic classes
    SEMANTIC_CLASSES = {
        'motion': ['ἔρχομαι', 'πορεύομαι', 'βαίνω', 'φεύγω', 'ἄγω', 'φέρω'],
        'speech': ['λέγω', 'φημί', 'λαλέω', 'κηρύσσω', 'ἀποκρίνομαι', 'καλέω'],
        'cognition': ['γινώσκω', 'οἶδα', 'νοέω', 'δοκέω', 'βούλομαι', 'θέλω'],
        'perception': ['ὁράω', 'βλέπω', 'ἀκούω', 'αἰσθάνομαι'],
        'emotion': ['χαίρω', 'λυπέω', 'φοβέω', 'θαυμάζω', 'ἀγαπάω'],
        'causation': ['ποιέω', 'δίδωμι', 'τίθημι', 'ἵστημι', 'πέμπω'],
    }

    # Stative verbs
    STATIVE_VERBS = [
        'εἰμί', 'ὑπάρχω', 'μένω', 'κεῖμαι', 'ἵσταμαι', 'ἧμαι',
        'οἶδα', 'ἔχω', 'δύναμαι', 'ὀφείλω',
    ]

    # Clause type vocabularies for encoding
    CLAUSE_TYPES = ['main', 'subordinate', 'participial', 'infinitival', 'relative']
    DEPRELS = ['root', 'nsubj', 'obj', 'iobj', 'obl', 'advmod', 'amod', 'nmod',
               'conj', 'cc', 'mark', 'advcl', 'acl', 'xcomp', 'ccomp', 'aux',
               'cop', 'det', 'case', 'punct', 'other']
    UPOS_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'AUX', 'CCONJ', 'SCONJ',
                 'DET', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'X', 'INTJ']
    INFO_STATUS = ['given', 'new', 'accessible']
    RHETORICAL_MODES = ['narrative', 'argument', 'exhortation', 'exposition']

    ARGUMENT_DEPRELS = {'nsubj', 'obj', 'iobj', 'obl', 'nsubj:pass'}
    REFERENT_UPOS = {'NOUN', 'PROPN', 'PRON'}

    def __init__(self, max_prev: int = 3, max_next: int = 2):
        """
        Initialize feature extractor.

        Args:
            max_prev: Maximum previous tokens to consider
            max_next: Maximum next tokens to consider
        """
        self.max_prev = max_prev
        self.max_next = max_next

        # Build index mappings
        self._build_index_maps()

        # Mention cache: sent_id -> list of MentionDescriptors
        self._mention_cache: Dict[str, List[MentionDescriptor]] = {}

    def _build_index_maps(self):
        """Build vocabulary to index mappings."""
        self.clause_type_to_idx = {ct: i for i, ct in enumerate(self.CLAUSE_TYPES)}
        self.deprel_to_idx = {dr: i for i, dr in enumerate(self.DEPRELS)}
        self.upos_to_idx = {pos: i for i, pos in enumerate(self.UPOS_TAGS)}
        self.info_status_to_idx = {s: i for i, s in enumerate(self.INFO_STATUS)}
        self.rhetorical_to_idx = {m: i for i, m in enumerate(self.RHETORICAL_MODES)}
        self.marker_to_idx = {m: i for i, m in enumerate(self.CLAUSE_MARKERS)}
        self.particle_to_idx = {p: i for i, p in enumerate(self.PARTICLES)}

    # ------------------------------------------------------------------
    # Cross-sentence referent tracking (Givón 1983, Prince 1981)
    # ------------------------------------------------------------------

    def _extract_mentions(self, sentence: Sentence) -> List[MentionDescriptor]:
        """
        Extract referent mentions (NOUN/PROPN/PRON in argument positions)
        from a sentence. Results are cached by sent_id.
        """
        sid = sentence.sent_id
        if sid in self._mention_cache:
            return self._mention_cache[sid]

        mentions: List[MentionDescriptor] = []
        for tok in sentence.word_tokens:
            if tok.upos not in self.REFERENT_UPOS:
                continue
            if tok.deprel not in self.ARGUMENT_DEPRELS:
                continue

            # Definiteness: check for DET child
            is_def = any(
                child.upos == "DET" for child in sentence.get_children(tok)
            )

            mentions.append(MentionDescriptor(
                lemma=tok.lemma,
                upos=tok.upos,
                deprel=tok.deprel,
                person=tok.get_feat("Person"),
                number=tok.get_feat("Number"),
                gender=tok.get_feat("Gender"),
                is_definite=is_def,
            ))

        self._mention_cache[sid] = mentions
        return mentions

    @staticmethod
    def _mentions_match(
        token_lemma: str,
        token_upos: str,
        token_person: str,
        token_number: str,
        token_gender: str,
        mention: MentionDescriptor,
    ) -> bool:
        """
        Heuristic coreference between a token and a prior mention.

        Rules:
          - NOUN/PROPN ↔ NOUN/PROPN: same lemma
          - PRON → any: Person + Number + Gender agree (wildcard _ matches)
          - NOUN/PROPN ↔ PRON: Number + Gender agree
        """
        def _agree(a: str, b: str) -> bool:
            return a == "_" or b == "_" or a == b

        tok_is_pron = token_upos == "PRON"
        men_is_pron = mention.upos == "PRON"

        if not tok_is_pron and not men_is_pron:
            # NOUN/PROPN ↔ NOUN/PROPN: same lemma
            return token_lemma == mention.lemma

        if tok_is_pron and not men_is_pron:
            # PRON → NOUN/PROPN: Person + Number + Gender agree
            return (
                _agree(token_person, mention.person)
                and _agree(token_number, mention.number)
                and _agree(token_gender, mention.gender)
            )

        if not tok_is_pron and men_is_pron:
            # NOUN/PROPN ↔ PRON: Number + Gender agree
            return (
                _agree(token_number, mention.number)
                and _agree(token_gender, mention.gender)
            )

        # PRON ↔ PRON: Person + Number + Gender agree
        return (
            _agree(token_person, mention.person)
            and _agree(token_number, mention.number)
            and _agree(token_gender, mention.gender)
        )

    def _compute_topic_continuity(
        self,
        token: ConlluToken,
        sentence: Sentence,
        prev_sentences: Optional[List[Sentence]],
    ) -> float:
        """
        Givón-style topic continuity score.

        For referent tokens (NOUN/PROPN/PRON in argument deprels):
          Search prev 1–3 sentences for a matching mention.
          Score = 1.0 / distance (1.0, 0.5, 0.33).

        For non-referent tokens:
          Jaccard overlap of noun-lemma sets between current and
          previous sentence.

        Returns 0.0 if no previous sentences.
        """
        if not prev_sentences:
            return 0.0

        # Referent tokens: distance-based score
        if (token.upos in self.REFERENT_UPOS
                and token.deprel in self.ARGUMENT_DEPRELS):
            t_lemma = token.lemma
            t_upos = token.upos
            t_person = token.get_feat("Person")
            t_number = token.get_feat("Number")
            t_gender = token.get_feat("Gender")

            for dist, prev_sent in enumerate(prev_sentences, start=1):
                prev_mentions = self._extract_mentions(prev_sent)
                for m in prev_mentions:
                    if self._mentions_match(
                        t_lemma, t_upos, t_person, t_number, t_gender, m
                    ):
                        return 1.0 / dist
            return 0.0

        # Non-referent tokens: Jaccard noun-lemma overlap with prev sentence
        cur_nouns = {
            t.lemma for t in sentence.word_tokens
            if t.upos in ("NOUN", "PROPN")
        }
        prev_nouns = {
            t.lemma for t in prev_sentences[0].word_tokens
            if t.upos in ("NOUN", "PROPN")
        }
        union = cur_nouns | prev_nouns
        if not union:
            return 0.0
        return len(cur_nouns & prev_nouns) / len(union)

    def _compute_information_status(
        self,
        token: ConlluToken,
        sentence: Sentence,
        prev_sentences: Optional[List[Sentence]],
    ) -> str:
        """
        Prince (1981) / Lambrecht (1994) information status.

        - "given":      referent matched in prev 1–3 sentences
        - "accessible": no match but PROPN, PRON, or has DET child (definite)
        - "new":        indefinite, no previous match
        """
        is_referent = (
            token.upos in self.REFERENT_UPOS
            and token.deprel in self.ARGUMENT_DEPRELS
        )

        # Check for match in previous sentences
        if is_referent and prev_sentences:
            t_lemma = token.lemma
            t_upos = token.upos
            t_person = token.get_feat("Person")
            t_number = token.get_feat("Number")
            t_gender = token.get_feat("Gender")

            for prev_sent in prev_sentences:
                for m in self._extract_mentions(prev_sent):
                    if self._mentions_match(
                        t_lemma, t_upos, t_person, t_number, t_gender, m
                    ):
                        return "given"

        # Accessible: PROPN, PRON, or definite (has DET child)
        if token.upos in ("PROPN", "PRON"):
            return "accessible"
        if any(child.upos == "DET" for child in sentence.get_children(token)):
            return "accessible"

        return "new"

    def extract(
        self,
        sentence: Sentence,
        position: int,
        prev_sentences: Optional[List[Sentence]] = None,
    ) -> HierarchicalFeatures:
        """
        Extract hierarchical features for a token at a given position.

        Args:
            sentence: Sentence object
            position: Token position (0-indexed into word_tokens)
            prev_sentences: Up to 3 preceding sentences (nearest first)
                for cross-sentence referent tracking

        Returns:
            HierarchicalFeatures object
        """
        tokens = sentence.word_tokens
        if position < 0 or position >= len(tokens):
            raise IndexError(f"Position {position} out of range for sentence with {len(tokens)} tokens")

        token = tokens[position]

        features = HierarchicalFeatures(
            token_position=position,
            token=token,
            semantic=self._extract_semantic(token, sentence),
            syntactic=self._extract_syntactic(token, sentence, position),
            discourse=self._extract_discourse(token, sentence, position, prev_sentences),
            context=self._extract_context(tokens, position, sentence=sentence),
        )

        return features

    def _extract_semantic(self, token: ConlluToken, sentence: Sentence) -> SemanticFeatures:
        """Extract Level 1 semantic features."""
        features = SemanticFeatures(lemma=token.lemma)

        # Determine Aktionsart
        features.aktionsart = self._get_aktionsart(token.lemma)
        features.aktionsart_idx = ['unknown', 'punctual', 'telic', 'atelic', 'durative'].index(
            features.aktionsart
        )

        # Determine argument structure (from dependency tree)
        features.argument_structure = self._get_argument_structure(token, sentence)
        features.arg_structure_idx = ['unknown', 'intransitive', 'transitive', 'ditransitive'].index(
            features.argument_structure
        )

        # Check if stative
        features.is_stative = token.lemma in self.STATIVE_VERBS
        features.stative_idx = 1 if features.is_stative else 0

        # Semantic class
        features.semantic_class = self._get_semantic_class(token.lemma)
        semantic_classes = ['unknown', 'motion', 'speech', 'cognition', 'perception', 'emotion', 'causation']
        features.semantic_class_idx = semantic_classes.index(features.semantic_class)

        return features

    def _get_aktionsart(self, lemma: str) -> str:
        """Determine Aktionsart category for a lemma."""
        for aktionsart, lemmas in self.AKTIONSART_PATTERNS.items():
            if lemma in lemmas:
                return aktionsart
        return "unknown"

    def _get_argument_structure(self, token: ConlluToken, sentence: Sentence) -> str:
        """Determine argument structure from dependency tree."""
        if token.head == "0":
            # This is the root - count objects
            children = sentence.get_children(token)
            has_obj = any(c.deprel == "obj" for c in children)
            has_iobj = any(c.deprel == "iobj" for c in children)

            if has_obj and has_iobj:
                return "ditransitive"
            elif has_obj:
                return "transitive"
            else:
                return "intransitive"

        return "unknown"

    def _get_semantic_class(self, lemma: str) -> str:
        """Determine semantic class for a lemma."""
        for sem_class, lemmas in self.SEMANTIC_CLASSES.items():
            if lemma in lemmas:
                return sem_class
        return "unknown"

    def _extract_syntactic(
        self,
        token: ConlluToken,
        sentence: Sentence,
        position: int
    ) -> SyntacticFeatures:
        """Extract Level 2 syntactic features."""
        features = SyntacticFeatures()

        # Clause type
        features.clause_type = sentence.get_clause_type(token)
        features.clause_type_idx = self.clause_type_to_idx.get(features.clause_type, 0)

        # Word order position
        features.word_order_position = position
        features.clause_length = self._compute_clause_length(token, sentence)
        if features.clause_length > 0:
            features.relative_position = position / features.clause_length

        # Tree depth and genitive chain depth
        features.tree_depth = self._compute_tree_depth(token, sentence)
        features.genitive_chain_depth = self._compute_genitive_chain_depth(token, sentence)

        # Dependency relation
        features.deprel = token.deprel
        features.deprel_idx = self.deprel_to_idx.get(token.deprel, len(self.DEPRELS) - 1)

        # Head POS
        head = sentence.get_head(token)
        if head:
            features.head_pos = head.upos
            features.head_pos_idx = self.upos_to_idx.get(head.upos, len(self.UPOS_TAGS) - 1)

        # Subject and object types
        if token.is_verb or token.deprel == "root":
            features.subject_type = self._get_subject_type(token, sentence)
            features.object_type = self._get_object_type(token, sentence)

        subj_types = ['unknown', 'nominal', 'pronominal', 'null']
        obj_types = ['unknown', 'nominal', 'pronominal', 'clausal', 'none']
        features.subject_type_idx = subj_types.index(features.subject_type)
        features.object_type_idx = obj_types.index(features.object_type)

        return features

    def _get_subject_type(self, token: ConlluToken, sentence: Sentence) -> str:
        """Determine subject type."""
        children = sentence.get_children(token)
        for child in children:
            if child.deprel == "nsubj":
                if child.upos == "PRON":
                    return "pronominal"
                elif child.upos in ["NOUN", "PROPN"]:
                    return "nominal"

        # Check for null subject (pro-drop)
        if token.is_verb and token.get_feat("Person") != "_":
            return "null"

        return "unknown"

    def _get_object_type(self, token: ConlluToken, sentence: Sentence) -> str:
        """Determine object type."""
        children = sentence.get_children(token)
        for child in children:
            if child.deprel == "obj":
                if child.upos == "PRON":
                    return "pronominal"
                elif child.upos in ["NOUN", "PROPN"]:
                    return "nominal"
            elif child.deprel in ["ccomp", "xcomp"]:
                return "clausal"

        return "none"

    def _compute_tree_depth(self, token: ConlluToken, sentence: Sentence) -> int:
        """Walk up dep tree from token to root, counting hops.

        Conj-aware: edges with deprel 'conj' do not increment the depth
        counter, so coordinated verbs ("Paul wrote and sent") receive
        the same syntactic depth as their head conjunct. This prevents
        UD's first-conjunct-as-head convention from penalizing coordinate
        main verbs in the downstream depth-weighting.

        Cycle-protected.
        """
        if token.head == "_" or token.head == "":
            return 0
        depth = 0
        visited = set()
        current = token
        while current is not None and current.head != "0":
            if current.id in visited:
                break  # cycle protection
            visited.add(current.id)
            # Do not count conj edges — coordinated elements are
            # syntactically parallel, not subordinate
            is_conj_edge = getattr(current, 'deprel', '') == 'conj'
            current = sentence.get_head(current)
            if not is_conj_edge:
                depth += 1
        return depth

    def _compute_clause_length(self, token: ConlluToken, sentence: Sentence) -> int:
        """Compute clause length by finding clause head and counting descendants."""
        if token.head == "_" or token.head == "":
            return len(sentence.word_tokens)

        # Walk up to clause head (root or clausal deprel boundary)
        clausal_deprels = {'advcl', 'ccomp', 'xcomp', 'acl', 'csubj'}
        clause_head = token
        visited = set()
        while clause_head is not None:
            if clause_head.id in visited:
                break
            visited.add(clause_head.id)
            if clause_head.head == "0" or clause_head.deprel in clausal_deprels:
                break
            parent = sentence.get_head(clause_head)
            if parent is None:
                break
            clause_head = parent

        # BFS count descendants within clause (stop at sub-clause boundaries)
        count = 1  # count the clause head itself
        queue = [clause_head]
        visited_bfs = {clause_head.id}
        while queue:
            node = queue.pop(0)
            children = sentence.get_children(node)
            for child in children:
                if child.id not in visited_bfs and child.deprel not in clausal_deprels:
                    visited_bfs.add(child.id)
                    count += 1
                    queue.append(child)

        return count

    def _compute_kai_rate(self, sentence: Sentence) -> float:
        """Fraction of CCONJ+cc tokens whose lemma is καί. Returns 0.0 if none."""
        cc_tokens = [t for t in sentence.word_tokens
                     if t.upos == "CCONJ" and t.deprel == "cc"]
        if not cc_tokens:
            return 0.0
        kai_count = sum(1 for t in cc_tokens if t.lemma == "καί")
        return kai_count / len(cc_tokens)

    def _compute_genitive_chain_depth(self, token: ConlluToken, sentence: Sentence) -> int:
        """Recursive walk down nmod children with Case=Gen. Returns max chain length."""
        children = sentence.get_children(token)
        max_depth = 0
        for child in children:
            if child.deprel == "nmod" and child.get_feat("Case") == "Gen":
                child_depth = 1 + self._compute_genitive_chain_depth(child, sentence)
                if child_depth > max_depth:
                    max_depth = child_depth
        return max_depth

    def _compute_article_pattern(
        self, tokens: List[ConlluToken], position: int, sentence: Sentence
    ) -> np.ndarray:
        """Binary vector [prev_noun_has_article, current_has_article, next_noun_has_article]."""
        pattern = np.zeros(3, dtype=np.float32)

        def _has_det_child(tok: ConlluToken) -> bool:
            """Check if token has a DET child."""
            for child in sentence.get_children(tok):
                if child.upos == "DET":
                    return True
            return False

        # Previous noun
        for i in range(position - 1, -1, -1):
            if tokens[i].upos in ("NOUN", "PROPN"):
                pattern[0] = 1.0 if _has_det_child(tokens[i]) else 0.0
                break

        # Current token
        pattern[1] = 1.0 if _has_det_child(tokens[position]) else 0.0

        # Next noun
        for i in range(position + 1, len(tokens)):
            if tokens[i].upos in ("NOUN", "PROPN"):
                pattern[2] = 1.0 if _has_det_child(tokens[i]) else 0.0
                break

        return pattern

    def _extract_discourse(
        self,
        token: ConlluToken,
        sentence: Sentence,
        position: int,
        prev_sentences: Optional[List[Sentence]] = None,
    ) -> DiscourseFeatures:
        """Extract Level 3 discourse features."""
        features = DiscourseFeatures()

        # Topic continuity — Givón (1983) distance-based referent tracking
        features.topic_continuity = self._compute_topic_continuity(
            token, sentence, prev_sentences
        )

        # Information status — Prince (1981) / Lambrecht (1994)
        features.information_status = self._compute_information_status(
            token, sentence, prev_sentences
        )

        features.information_status_idx = self.info_status_to_idx.get(
            features.information_status, 1
        )

        # Rhetorical mode (simplified heuristic based on verb forms)
        features.rhetorical_mode = self._infer_rhetorical_mode(sentence)
        features.rhetorical_mode_idx = self.rhetorical_to_idx.get(
            features.rhetorical_mode, 0
        )

        # Topic/focus detection
        if token.deprel == "nsubj" and position < 3:
            features.is_topic = True
        if token.deprel in ["obj", "obl"] and position > len(sentence) // 2:
            features.is_focus = True

        # Paratactic καί rate
        features.kai_rate = self._compute_kai_rate(sentence)

        return features

    def _infer_rhetorical_mode(self, sentence: Sentence) -> str:
        """Infer rhetorical mode from sentence characteristics."""
        # Check for imperatives → exhortation
        for token in sentence.word_tokens:
            if token.get_feat("Mood") == "Imp":
                return "exhortation"

        # Check for connectors suggesting argumentation
        for token in sentence.word_tokens:
            if token.form.lower() in ['γάρ', 'οὖν', 'διότι', 'ὥστε', 'εἰ', 'ἐάν']:
                return "argument"

        # Check for aorist narrative chains
        aorist_count = sum(1 for t in sentence.word_tokens if t.get_feat("Tense") in ("Past", "Aor"))
        if aorist_count > len(sentence) // 3:
            return "narrative"

        return "exposition"

    def _extract_context(
        self,
        tokens: List[ConlluToken],
        position: int,
        sentence: Sentence = None,
    ) -> MicroContextFeatures:
        """Extract Level 4 micro-context features."""
        features = MicroContextFeatures()

        # Previous tokens
        for i in range(self.max_prev):
            prev_idx = position - i - 1
            if prev_idx >= 0:
                prev_token = tokens[prev_idx]
                features.prev_forms.append(prev_token.form)
                features.prev_lemmas.append(prev_token.lemma)
                features.prev_pos.append(prev_token.upos)
                features.prev_morph_codes.append(
                    MorphologyNormalizer.token_to_proiel(prev_token)
                )
            else:
                # Padding
                features.prev_forms.append("<PAD>")
                features.prev_lemmas.append("<PAD>")
                features.prev_pos.append("<PAD>")
                features.prev_morph_codes.append("-" * 10)

        # Next tokens
        for i in range(self.max_next):
            next_idx = position + i + 1
            if next_idx < len(tokens):
                next_token = tokens[next_idx]
                features.next_forms.append(next_token.form)
                features.next_lemmas.append(next_token.lemma)
                features.next_pos.append(next_token.upos)
                features.next_morph_codes.append(
                    MorphologyNormalizer.token_to_proiel(next_token)
                )
            else:
                # Padding
                features.next_forms.append("<PAD>")
                features.next_lemmas.append("<PAD>")
                features.next_pos.append("<PAD>")
                features.next_morph_codes.append("-" * 10)

        # Clause markers in context window
        context_window = []
        for i in range(max(0, position - 3), min(len(tokens), position + 3)):
            context_window.append(tokens[i].form.lower())

        for marker in self.CLAUSE_MARKERS:
            if marker.lower() in context_window:
                features.clause_markers.append(marker)

        # Particles in context window
        for particle in self.PARTICLES:
            if particle.lower() in context_window:
                features.particles.append(particle)

        # Position flags
        features.is_clause_initial = position == 0
        features.is_clause_final = position == len(tokens) - 1

        if position > 0:
            prev_form = tokens[position - 1].form.lower()
            features.follows_particle = prev_form in [p.lower() for p in self.PARTICLES]

        if position < len(tokens) - 1:
            next_form = tokens[position + 1].form.lower()
            features.precedes_particle = next_form in [p.lower() for p in self.PARTICLES]

        # Create numeric vectors
        features.clause_marker_vector = np.zeros(len(self.CLAUSE_MARKERS))
        for marker in features.clause_markers:
            idx = self.marker_to_idx.get(marker, -1)
            if idx >= 0:
                features.clause_marker_vector[idx] = 1.0

        features.particle_vector = np.zeros(len(self.PARTICLES))
        for particle in features.particles:
            idx = self.particle_to_idx.get(particle, -1)
            if idx >= 0:
                features.particle_vector[idx] = 1.0

        # Article presence pattern
        if sentence is not None:
            features.article_pattern = self._compute_article_pattern(
                tokens, position, sentence
            )

        return features

    def extract_batch(
        self,
        sentence: Sentence,
        positions: Optional[List[int]] = None,
        prev_sentences: Optional[List[Sentence]] = None,
    ) -> List[HierarchicalFeatures]:
        """
        Extract features for multiple positions in a sentence.

        Args:
            sentence: Sentence object
            positions: List of positions (default: all positions)
            prev_sentences: Up to 3 preceding sentences (nearest first)

        Returns:
            List of HierarchicalFeatures objects
        """
        if positions is None:
            positions = list(range(len(sentence.word_tokens)))

        return [self.extract(sentence, pos, prev_sentences) for pos in positions]

    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of all feature types."""
        return {
            "clause_types": len(self.CLAUSE_TYPES),
            "deprels": len(self.DEPRELS),
            "upos_tags": len(self.UPOS_TAGS),
            "info_status": len(self.INFO_STATUS),
            "rhetorical_modes": len(self.RHETORICAL_MODES),
            "clause_markers": len(self.CLAUSE_MARKERS),
            "particles": len(self.PARTICLES),
            "aktionsart": 5,
            "arg_structure": 4,
            "semantic_class": 7,
        }
