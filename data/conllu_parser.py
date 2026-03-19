"""
CoNLL-U format parser for Ancient Greek corpora.

Parses CoNLL-U files into structured Sentence and Corpus objects,
handling the full Universal Dependencies annotation scheme.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Union
import re


@dataclass
class ConlluToken:
    """A single token from CoNLL-U format."""

    id: str                          # Token ID (can be range like "1-2")
    form: str                        # Surface form (Greek text)
    lemma: str                       # Dictionary form
    upos: str                        # Universal POS (NOUN, VERB, ADJ, etc.)
    xpos: str                        # Language-specific tag
    feats: Dict[str, str]            # Morphological features
    head: str                        # Dependency head ID
    deprel: str                      # Dependency relation
    deps: str = "_"                  # Enhanced dependencies
    misc: str = "_"                  # Miscellaneous annotations

    @property
    def is_multiword(self) -> bool:
        """Check if this is a multiword token (range ID like 1-2)."""
        return "-" in self.id

    @property
    def is_empty(self) -> bool:
        """Check if this is an empty node (decimal ID like 1.1)."""
        return "." in self.id

    @property
    def int_id(self) -> int:
        """Get integer ID (for regular tokens only)."""
        if self.is_multiword or self.is_empty:
            return -1
        return int(self.id)

    def get_feat(self, feat_name: str, default: str = "_") -> str:
        """Get a specific feature value."""
        return self.feats.get(feat_name, default)

    @property
    def tense(self) -> str:
        return self.get_feat("Tense")

    @property
    def voice(self) -> str:
        return self.get_feat("Voice")

    @property
    def mood(self) -> str:
        return self.get_feat("Mood")

    @property
    def case(self) -> str:
        return self.get_feat("Case")

    @property
    def number(self) -> str:
        return self.get_feat("Number")

    @property
    def gender(self) -> str:
        return self.get_feat("Gender")

    @property
    def person(self) -> str:
        return self.get_feat("Person")

    @property
    def degree(self) -> str:
        return self.get_feat("Degree")

    @property
    def prontype(self) -> str:
        return self.get_feat("PronType")

    @property
    def definite(self) -> str:
        return self.get_feat("Definite")

    @property
    def misc_dict(self) -> Dict[str, str]:
        """Parse the MISC column into a dict (cached on first access)."""
        if not hasattr(self, "_misc_dict"):
            from . import conllu_parser as _mod
            self._misc_dict = _mod.ConlluParser.parse_misc(self.misc)
        return self._misc_dict

    @property
    def provenance(self) -> str:
        """Provenance tag from MISC column (e.g. 'constrained', 'fuzzy', 'unconstrained')."""
        return self.misc_dict.get("provenance", "_")

    @property
    def confidence(self) -> float:
        """Confidence score from MISC column (0.0-1.0)."""
        val = self.misc_dict.get("confidence", "1.0")
        try:
            return float(val)
        except (ValueError, TypeError):
            return 1.0

    @property
    def is_verb(self) -> bool:
        return self.upos == "VERB" or self.upos == "AUX"

    @property
    def is_noun(self) -> bool:
        return self.upos == "NOUN" or self.upos == "PROPN"

    @property
    def is_adjective(self) -> bool:
        return self.upos == "ADJ"

    @property
    def is_pronoun(self) -> bool:
        return self.upos == "PRON"

    def __repr__(self) -> str:
        return f"ConlluToken({self.id}: {self.form}/{self.lemma} [{self.upos}])"


@dataclass
class Sentence:
    """A sentence containing multiple tokens."""

    tokens: List[ConlluToken]
    sent_id: str = ""
    text: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)

    # Author information for author-conditioned training
    author_id: str = ""      # e.g., "PHILO_JUDAEUS", "SEPTUAGINTA"
    tradition: str = ""      # "jewish" or "gentile"

    @property
    def tradition_idx(self) -> int:
        """Get numeric index for tradition (0=jewish, 1=gentile)."""
        return 0 if self.tradition == "jewish" else 1

    @property
    def word_tokens(self) -> List[ConlluToken]:
        """Get only regular word tokens (not multiword or empty)."""
        return [t for t in self.tokens if not t.is_multiword and not t.is_empty]

    @property
    def verbs(self) -> List[ConlluToken]:
        """Get all verbs in the sentence."""
        return [t for t in self.word_tokens if t.is_verb]

    @property
    def nouns(self) -> List[ConlluToken]:
        """Get all nouns in the sentence."""
        return [t for t in self.word_tokens if t.is_noun]

    def get_token_by_id(self, token_id: str) -> Optional[ConlluToken]:
        """Get a token by its ID."""
        for token in self.tokens:
            if token.id == token_id:
                return token
        return None

    def get_head(self, token: ConlluToken) -> Optional[ConlluToken]:
        """Get the dependency head of a token."""
        if token.head == "0":
            return None  # Root
        return self.get_token_by_id(token.head)

    def get_children(self, token: ConlluToken) -> List[ConlluToken]:
        """Get all dependents of a token."""
        return [t for t in self.word_tokens if t.head == token.id]

    def get_root(self) -> Optional[ConlluToken]:
        """Get the root token of the sentence."""
        for token in self.word_tokens:
            if token.head == "0":
                return token
        return None

    def get_clause_type(self, token: ConlluToken) -> str:
        """
        Determine clause type for a token based on dependency structure.
        Returns: 'main', 'subordinate', 'participial', 'infinitival', 'relative'
        """
        # Check if in participial clause
        if token.upos == "VERB" and token.get_feat("VerbForm") == "Part":
            return "participial"

        # Check if in infinitival clause
        if token.upos == "VERB" and token.get_feat("VerbForm") == "Inf":
            return "infinitival"

        # Check for subordinating conjunctions
        if token.deprel in ["advcl", "csubj", "ccomp"]:
            return "subordinate"

        # Check for relative clauses
        if token.deprel == "acl:relcl":
            return "relative"

        # Check if this token is dependent on a subordinate clause marker
        head = self.get_head(token)
        if head and head.deprel in ["mark", "advcl", "ccomp"]:
            return "subordinate"

        return "main"

    def __len__(self) -> int:
        return len(self.word_tokens)

    def __iter__(self) -> Iterator[ConlluToken]:
        return iter(self.word_tokens)

    def __getitem__(self, idx: int) -> ConlluToken:
        return self.word_tokens[idx]

    def __repr__(self) -> str:
        return f"Sentence({self.sent_id}: {len(self.tokens)} tokens)"


@dataclass
class Corpus:
    """A collection of sentences."""

    sentences: List[Sentence]
    name: str = ""
    source_path: Optional[Path] = None
    corpus_type: str = ""  # "jewish" or "gentile"

    # Author information (optional, for single-author corpora)
    author_id: str = ""
    tradition: str = ""

    @property
    def tradition_idx(self) -> int:
        """Get numeric index for tradition (0=jewish, 1=gentile)."""
        return 0 if self.tradition == "jewish" else 1

    def tag_with_author(self, author_id: str, tradition: str) -> "Corpus":
        """
        Tag all sentences in the corpus with author information.

        Args:
            author_id: Author identifier
            tradition: "jewish" or "gentile"

        Returns:
            Self for method chaining
        """
        self.author_id = author_id
        self.tradition = tradition
        for sentence in self.sentences:
            sentence.author_id = author_id
            sentence.tradition = tradition
        return self

    @property
    def total_tokens(self) -> int:
        """Total word tokens across all sentences."""
        return sum(len(s) for s in self.sentences)

    @property
    def vocabulary(self) -> set:
        """Unique lemmas in the corpus."""
        vocab = set()
        for sent in self.sentences:
            for token in sent:
                vocab.add(token.lemma)
        return vocab

    def get_lemma_forms(self, lemma: str) -> Dict[str, int]:
        """Get all surface forms for a lemma with counts."""
        forms: Dict[str, int] = {}
        for sent in self.sentences:
            for token in sent:
                if token.lemma == lemma:
                    forms[token.form] = forms.get(token.form, 0) + 1
        return forms

    def get_tokens_by_lemma(self, lemma: str) -> List[ConlluToken]:
        """Get all tokens with a specific lemma."""
        tokens = []
        for sent in self.sentences:
            for token in sent:
                if token.lemma == lemma:
                    tokens.append(token)
        return tokens

    def filter_by_pos(self, pos: str) -> "Corpus":
        """Create a filtered corpus containing only tokens of a specific POS."""
        filtered_sents = []
        for sent in self.sentences:
            filtered_tokens = [t for t in sent.tokens if t.upos == pos]
            if filtered_tokens:
                filtered_sents.append(Sentence(
                    tokens=filtered_tokens,
                    sent_id=sent.sent_id,
                    text=sent.text,
                    metadata=sent.metadata,
                    author_id=sent.author_id,
                    tradition=sent.tradition,
                ))
        return Corpus(
            sentences=filtered_sents,
            name=f"{self.name}_{pos}",
            source_path=self.source_path,
            corpus_type=self.corpus_type,
            author_id=self.author_id,
            tradition=self.tradition,
        )

    def __len__(self) -> int:
        return len(self.sentences)

    def __iter__(self) -> Iterator[Sentence]:
        return iter(self.sentences)

    def __repr__(self) -> str:
        return f"Corpus({self.name}: {len(self.sentences)} sentences, {self.total_tokens} tokens)"


class ConlluParser:
    """Parser for CoNLL-U format files."""

    # Canonical casing for feature keys that appear with inconsistent
    # capitalisation across taggers (e.g. ``Prontype`` → ``PronType``).
    _KEY_CANONICAL = {
        "prontype": "PronType",
        "verbform": "VerbForm",
        "definite": "Definite",
        "numtype": "NumType",
        "reflex": "Reflex",
        "poss": "Poss",
        "abbr": "Abbr",
        "foreign": "Foreign",
        "typo": "Typo",
    }

    @staticmethod
    def parse_features(feat_str: str) -> Dict[str, str]:
        """Parse feature string like 'Case=Nom|Number=Sing' into dict.

        Feature keys are case-normalised so that e.g. ``Prontype`` and
        ``PronType`` both map to ``PronType``.
        """
        if feat_str == "_" or not feat_str:
            return {}

        feats = {}
        for pair in feat_str.split("|"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                # Case-normalise known keys
                canonical = ConlluParser._KEY_CANONICAL.get(key.lower())
                if canonical is not None:
                    key = canonical
                feats[key] = value
        return feats

    @staticmethod
    def parse_misc(misc_str: str) -> Dict[str, str]:
        """Parse MISC column (same pipe-separated key=value format)."""
        if misc_str == "_" or not misc_str:
            return {}
        result = {}
        for pair in misc_str.split("|"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                result[key] = value
        return result

    @staticmethod
    def parse_token(line: str) -> Optional[ConlluToken]:
        """Parse a single CoNLL-U token line."""
        if not line or line.startswith("#"):
            return None

        parts = line.strip().split("\t")
        if len(parts) < 10:
            # Try space-separated (some older files)
            parts = line.strip().split()
            if len(parts) < 10:
                return None

        return ConlluToken(
            id=parts[0],
            form=parts[1],
            lemma=parts[2],
            upos=parts[3],
            xpos=parts[4] if parts[4] != "_" else parts[3],
            feats=ConlluParser.parse_features(parts[5]),
            head=parts[6],
            deprel=parts[7],
            deps=parts[8] if len(parts) > 8 else "_",
            misc=parts[9] if len(parts) > 9 else "_"
        )

    def parse_sentence(self, lines: List[str]) -> Sentence:
        """Parse lines into a Sentence object."""
        tokens = []
        metadata = {}
        sent_id = ""
        text = ""

        for line in lines:
            if line.startswith("# sent_id"):
                sent_id = line.split("=", 1)[1].strip() if "=" in line else ""
            elif line.startswith("# text"):
                text = line.split("=", 1)[1].strip() if "=" in line else ""
            elif line.startswith("#"):
                # Other metadata
                if "=" in line:
                    key, value = line[1:].strip().split("=", 1)
                    metadata[key.strip()] = value.strip()
            else:
                token = self.parse_token(line)
                if token:
                    tokens.append(token)

        return Sentence(
            tokens=tokens,
            sent_id=sent_id,
            text=text,
            metadata=metadata
        )

    def parse_file(self, path: Union[str, Path]) -> List[Sentence]:
        """Parse a single CoNLL-U file."""
        path = Path(path)
        sentences = []

        with open(path, "r", encoding="utf-8") as f:
            current_lines = []

            for line in f:
                line = line.rstrip("\n")

                if not line:
                    # Empty line = end of sentence
                    if current_lines:
                        sent = self.parse_sentence(current_lines)
                        if sent.tokens:
                            sentences.append(sent)
                        current_lines = []
                else:
                    current_lines.append(line)

            # Handle last sentence if file doesn't end with blank line
            if current_lines:
                sent = self.parse_sentence(current_lines)
                if sent.tokens:
                    sentences.append(sent)

        return sentences

    def parse_directory(
        self,
        path: Union[str, Path],
        recursive: bool = True,
        pattern: str = "*.conllu",
        corpus_type: str = ""
    ) -> Corpus:
        """
        Parse all CoNLL-U files in a directory.

        Args:
            path: Directory path
            recursive: If True, search subdirectories
            pattern: Glob pattern for files
            corpus_type: "jewish" or "gentile" for corpus labeling

        Returns:
            Corpus object containing all sentences
        """
        path = Path(path)
        all_sentences = []

        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))

        # Also check for .conll extension
        if recursive:
            files.extend(list(path.rglob("*.conll")))
        else:
            files.extend(list(path.glob("*.conll")))

        for file_path in sorted(files):
            try:
                sentences = self.parse_file(file_path)
                # Tag sentences with source file for author inference
                for sent in sentences:
                    sent.metadata['source_file'] = str(file_path)
                all_sentences.extend(sentences)
            except Exception as e:
                print(f"Warning: Could not parse {file_path}: {e}")
                continue

        return Corpus(
            sentences=all_sentences,
            name=path.name,
            source_path=path,
            corpus_type=corpus_type
        )

    def parse_tagged_corpus(
        self,
        root: Union[str, Path],
    ) -> List[Corpus]:
        """
        Parse a ``TaggedCorpus/`` directory tree.

        Expected layout::

            root/
              jewish/
                AUTHOR_NAME/
                  WORK.conllu
              gentile/
                AUTHOR_NAME/
                  WORK.conllu

        Returns one :class:`Corpus` per tradition, with every sentence
        tagged with ``author_id`` and ``tradition``.
        """
        root = Path(root)
        corpora: List[Corpus] = []

        for tradition in ("jewish", "gentile"):
            tradition_dir = root / tradition
            if not tradition_dir.is_dir():
                continue

            all_sentences: List[Sentence] = []
            for author_dir in sorted(tradition_dir.iterdir()):
                if not author_dir.is_dir():
                    continue
                author_id = author_dir.name
                for conllu_file in sorted(author_dir.glob("*.conllu")):
                    try:
                        sentences = self.parse_file(conllu_file)
                        for sent in sentences:
                            sent.author_id = author_id
                            sent.tradition = tradition
                            sent.metadata["source_file"] = str(conllu_file)
                        all_sentences.extend(sentences)
                    except Exception as e:
                        print(f"Warning: Could not parse {conllu_file}: {e}")

            corpus = Corpus(
                sentences=all_sentences,
                name=tradition,
                source_path=tradition_dir,
                corpus_type=tradition,
                tradition=tradition,
            )
            corpora.append(corpus)

        return corpora

    def parse_string(self, text: str) -> List[Sentence]:
        """Parse CoNLL-U format from a string."""
        sentences = []
        current_lines = []

        for line in text.split("\n"):
            line = line.rstrip()

            if not line:
                if current_lines:
                    sent = self.parse_sentence(current_lines)
                    if sent.tokens:
                        sentences.append(sent)
                    current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sent = self.parse_sentence(current_lines)
            if sent.tokens:
                sentences.append(sent)

        return sentences


def merge_corpora(*corpora: Corpus, name: str = "merged") -> Corpus:
    """Merge multiple corpora into one.

    Note: Author information is preserved at sentence level.
    Corpus-level author info is taken from the first corpus if all match.
    """
    all_sentences = []
    for corpus in corpora:
        all_sentences.extend(corpus.sentences)

    # Determine corpus-level author info
    # If all corpora have the same tradition, use it; otherwise leave empty
    traditions = set(c.tradition for c in corpora if c.tradition)
    tradition = traditions.pop() if len(traditions) == 1 else ""

    return Corpus(
        sentences=all_sentences,
        name=name,
        corpus_type=corpora[0].corpus_type if corpora else "",
        tradition=tradition,
    )


def split_corpus(
    corpus: Corpus,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> tuple:
    """Split corpus into train/val/test sets."""
    n = len(corpus.sentences)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_sents = corpus.sentences[:train_end]
    val_sents = corpus.sentences[train_end:val_end]
    test_sents = corpus.sentences[val_end:]

    return (
        Corpus(
            sentences=train_sents,
            name=f"{corpus.name}_train",
            corpus_type=corpus.corpus_type,
            author_id=corpus.author_id,
            tradition=corpus.tradition,
        ),
        Corpus(
            sentences=val_sents,
            name=f"{corpus.name}_val",
            corpus_type=corpus.corpus_type,
            author_id=corpus.author_id,
            tradition=corpus.tradition,
        ),
        Corpus(
            sentences=test_sents,
            name=f"{corpus.name}_test",
            corpus_type=corpus.corpus_type,
            author_id=corpus.author_id,
            tradition=corpus.tradition,
        ),
    )
