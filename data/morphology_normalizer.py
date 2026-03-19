"""
Morphology normalizer for converting between PROIEL and CoNLL-U formats.

PROIEL uses a 10-character positional code:
Position 1: POS (N=noun, V=verb, A=adjective, etc.)
Position 2: Person (1,2,3,-)
Position 3: Number (s=sing, p=plur, d=dual, -)
Position 4: Tense (p=pres, i=imperf, r=perf, s=pluperf, a=aorist, u=fut, l=futperf, f=futperf)
Position 5: Mood (i=ind, s=subj, m=imper, o=opt, n=inf, p=part, d=gerund, g=gerundive)
Position 6: Voice (a=act, m=mid, p=pass, e=mid-pass)
Position 7: Gender (m=masc, f=fem, n=neut, c=common)
Position 8: Case (n=nom, a=acc, g=gen, d=dat, v=voc, o=oblique)
Position 9: Degree (p=pos, c=comp, s=super)
Position 10: Strength (w=weak/indef, s=strong/def)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from .conllu_parser import ConlluToken


@dataclass
class MorphFeatures:
    """Structured morphological features."""

    pos: str = ""          # Part of speech
    person: str = ""       # 1, 2, 3
    number: str = ""       # Sing, Plur, Dual
    tense: str = ""        # Pres, Imp, Aor, Perf, Plup, Fut
    mood: str = ""         # Ind, Sub, Imp, Opt, Inf, Part
    voice: str = ""        # Act, Mid, Pass
    gender: str = ""       # Masc, Fem, Neut
    case: str = ""         # Nom, Acc, Gen, Dat, Voc
    degree: str = ""       # Pos, Cmp, Sup
    definiteness: str = "" # Ind, Def
    prontype: str = ""     # Prs, Dem, Rel, Int, Ind, Rcp, Tot, Neg

    def to_proiel_code(self) -> str:
        """Convert to 10-character PROIEL code."""
        return MorphologyNormalizer.features_to_proiel(self)

    def to_conllu_feats(self) -> Dict[str, str]:
        """Convert to CoNLL-U feature dict."""
        return MorphologyNormalizer.features_to_conllu(self)


class MorphologyNormalizer:
    """
    Bidirectional converter between PROIEL codes and CoNLL-U features.
    """

    # PROIEL to CoNLL-U mappings
    PROIEL_POS_TO_UPOS = {
        "N": "NOUN",
        "V": "VERB",
        "A": "ADJ",
        "D": "ADV",
        "C": "CCONJ",
        "R": "ADP",
        "P": "PRON",
        "M": "NUM",
        "I": "INTJ",
        "G": "SCONJ",
        "X": "X",
        "-": "_",
    }

    UPOS_TO_PROIEL_POS = {v: k for k, v in PROIEL_POS_TO_UPOS.items()}
    UPOS_TO_PROIEL_POS["PROPN"] = "N"
    UPOS_TO_PROIEL_POS["AUX"] = "V"
    UPOS_TO_PROIEL_POS["DET"] = "A"
    UPOS_TO_PROIEL_POS["PART"] = "X"
    UPOS_TO_PROIEL_POS["PUNCT"] = "X"

    # Person mappings
    PROIEL_PERSON = {"1": "1", "2": "2", "3": "3", "-": "_"}
    CONLLU_PERSON = {"1": "1", "2": "2", "3": "3", "_": "-"}

    # Number mappings
    PROIEL_NUMBER = {"s": "Sing", "p": "Plur", "d": "Dual", "-": "_"}
    CONLLU_NUMBER = {"Sing": "s", "Plur": "p", "Dual": "d", "_": "-"}

    # Tense mappings - Greek-specific
    # Internal representation uses the CoNLL-U values from the corpus:
    #   Aor (aorist), Imp (imperfect), Perf (perfect), Plup (pluperfect),
    #   Pres (present), Fut (future)
    PROIEL_TENSE = {
        "p": "Pres",      # Present
        "i": "Imp",       # Imperfect
        "r": "Perf",      # Perfect
        "s": "Plup",      # Pluperfect
        "a": "Aor",       # Aorist
        "u": "Fut",       # Future
        "l": "FutPerf",   # Future perfect — now a distinct class (v2)
        "f": "FutPerf",   # Future perfect variant
        "-": "_",
    }

    CONLLU_TENSE = {
        "Pres": "p",
        "Imp": "i",
        "Imperf": "i",    # Alternate spelling seen in some taggers
        "Perf": "r",
        "Plup": "s",
        "Aor": "a",
        "Fut": "u",
        "FutPerf": "l",   # Future perfect — now distinct (v2)
        # Legacy values for backward compatibility
        "Past": "a",
        "Pqp": "s",
        "_": "-",
    }

    # Extended tense names for display
    TENSE_NAMES = {
        "p": "present",
        "i": "imperfect",
        "r": "perfect",
        "s": "pluperfect",
        "a": "aorist",
        "u": "future",
        "l": "future_perfect",
        "f": "future_perfect",
    }

    # Mood mappings
    PROIEL_MOOD = {
        "i": "Ind",       # Indicative
        "s": "Sub",       # Subjunctive
        "m": "Imp",       # Imperative
        "o": "Opt",       # Optative
        "n": "Inf",       # Infinitive
        "p": "Part",      # Participle
        "d": "Ger",       # Gerund
        "g": "Gdv",       # Gerundive
        "-": "_",
    }

    CONLLU_MOOD = {
        "Ind": "i",
        "Sub": "s",
        "Imp": "m",
        "Opt": "o",
        "_": "-",
    }

    CONLLU_VERBFORM = {
        "Inf": "n",
        "Part": "p",
        "Ger": "d",
        "Gdv": "g",
    }

    # Voice mappings
    PROIEL_VOICE = {
        "a": "Act",
        "m": "Mid",
        "p": "Pass",
        "e": "MidPass",   # Middle-passive (distinct from pure middle)
        "-": "_",
    }

    CONLLU_VOICE = {
        "Act": "a",
        "Mid": "m",
        "Pass": "p",
        "MidPass": "e",
        "_": "-",
    }

    # Gender mappings
    PROIEL_GENDER = {
        "m": "Masc",
        "f": "Fem",
        "n": "Neut",
        "c": "Com",       # Common gender
        "-": "_",
    }

    CONLLU_GENDER = {
        "Masc": "m",
        "Fem": "f",
        "Neut": "n",
        "Com": "c",
        "_": "-",
    }

    # Case mappings
    PROIEL_CASE = {
        "n": "Nom",
        "a": "Acc",
        "g": "Gen",
        "d": "Dat",
        "v": "Voc",
        "o": "Dat",       # Oblique → Dat (closest)
        "-": "_",
    }

    CONLLU_CASE = {
        "Nom": "n",
        "Acc": "a",
        "Gen": "g",
        "Dat": "d",
        "Voc": "v",
        "_": "-",
    }

    # Degree mappings
    PROIEL_DEGREE = {
        "p": "Pos",
        "c": "Cmp",
        "s": "Sup",
        "-": "_",
    }

    CONLLU_DEGREE = {
        "Pos": "p",
        "Cmp": "c",
        "Sup": "s",
        "_": "-",
    }

    @classmethod
    def proiel_to_features(cls, proiel_code: str) -> MorphFeatures:
        """
        Convert PROIEL 10-character code to structured features.

        Args:
            proiel_code: 10-character positional code

        Returns:
            MorphFeatures object
        """
        # Pad to 10 characters if needed
        proiel_code = proiel_code.ljust(10, "-")

        features = MorphFeatures()

        # Position 1: POS
        features.pos = cls.PROIEL_POS_TO_UPOS.get(proiel_code[0], "_")

        # Position 2: Person
        features.person = cls.PROIEL_PERSON.get(proiel_code[1], "_")

        # Position 3: Number
        features.number = cls.PROIEL_NUMBER.get(proiel_code[2], "_")

        # Position 4: Tense
        features.tense = cls.PROIEL_TENSE.get(proiel_code[3], "_")

        # Position 5: Mood
        features.mood = cls.PROIEL_MOOD.get(proiel_code[4], "_")

        # Position 6: Voice
        features.voice = cls.PROIEL_VOICE.get(proiel_code[5], "_")

        # Position 7: Gender
        features.gender = cls.PROIEL_GENDER.get(proiel_code[6], "_")

        # Position 8: Case
        features.case = cls.PROIEL_CASE.get(proiel_code[7], "_")

        # Position 9: Degree
        features.degree = cls.PROIEL_DEGREE.get(proiel_code[8], "_")

        # Position 10: Definiteness
        if proiel_code[9] == "w":
            features.definiteness = "Ind"
        elif proiel_code[9] == "s":
            features.definiteness = "Def"
        else:
            features.definiteness = "_"

        return features

    @classmethod
    def features_to_proiel(cls, features: MorphFeatures) -> str:
        """
        Convert MorphFeatures to PROIEL 10-character code.

        Args:
            features: MorphFeatures object

        Returns:
            10-character PROIEL code
        """
        code = []

        # Position 1: POS
        code.append(cls.UPOS_TO_PROIEL_POS.get(features.pos, "-"))

        # Position 2: Person
        code.append(cls.CONLLU_PERSON.get(features.person, "-"))

        # Position 3: Number
        code.append(cls.CONLLU_NUMBER.get(features.number, "-"))

        # Position 4: Tense
        code.append(cls.CONLLU_TENSE.get(features.tense, "-"))

        # Position 5: Mood
        mood_code = cls.CONLLU_MOOD.get(features.mood, "-")
        if mood_code == "-":
            mood_code = cls.CONLLU_VERBFORM.get(features.mood, "-")
        code.append(mood_code)

        # Position 6: Voice
        code.append(cls.CONLLU_VOICE.get(features.voice, "-"))

        # Position 7: Gender
        code.append(cls.CONLLU_GENDER.get(features.gender, "-"))

        # Position 8: Case
        code.append(cls.CONLLU_CASE.get(features.case, "-"))

        # Position 9: Degree
        code.append(cls.CONLLU_DEGREE.get(features.degree, "-"))

        # Position 10: Definiteness
        if features.definiteness == "Ind":
            code.append("w")
        elif features.definiteness == "Def":
            code.append("s")
        else:
            code.append("-")

        return "".join(code)

    @classmethod
    def conllu_to_features(cls, token: ConlluToken) -> MorphFeatures:
        """
        Convert CoNLL-U token to MorphFeatures.

        Args:
            token: ConlluToken object

        Returns:
            MorphFeatures object
        """
        features = MorphFeatures()
        features.pos = token.upos

        features.person = token.feats.get("Person", "_")
        features.number = token.feats.get("Number", "_")
        features.gender = token.feats.get("Gender", "_")
        features.case = token.feats.get("Case", "_")
        features.degree = token.feats.get("Degree", "_")
        features.definiteness = token.feats.get("Definite", "_")

        # Tense — read directly; normalise legacy "Past" → "Aor"
        raw_tense = token.feats.get("Tense", "_")
        _tense_normalize = {"Past": "Aor", "Pqp": "Plup", "Imperf": "Imp"}
        features.tense = _tense_normalize.get(raw_tense, raw_tense)

        # Voice
        features.voice = token.feats.get("Voice", "_")

        # Mood - handle VerbForm for participles/infinitives
        features.mood = token.feats.get("Mood", "_")
        verbform = token.feats.get("VerbForm", "_")
        if verbform in ["Inf", "Part", "Ger", "Gdv"]:
            features.mood = verbform

        # PronType — normalise to canonical form
        raw_prontype = token.feats.get("PronType", "_")
        features.prontype = cls.normalize_prontype(raw_prontype)

        return features

    @classmethod
    def features_to_conllu(cls, features: MorphFeatures) -> Dict[str, str]:
        """
        Convert MorphFeatures to CoNLL-U feature dict.

        Args:
            features: MorphFeatures object

        Returns:
            Dictionary of CoNLL-U features
        """
        feats = {}

        if features.person != "_":
            feats["Person"] = features.person
        if features.number != "_":
            feats["Number"] = features.number
        if features.gender != "_":
            feats["Gender"] = features.gender
        if features.case != "_":
            feats["Case"] = features.case

        # Tense — output the CoNLL-U value directly
        if features.tense != "_":
            # Map legacy internal names to CoNLL-U values
            _legacy_tense = {"Past": "Aor", "Pqp": "Plup"}
            feats["Tense"] = _legacy_tense.get(features.tense, features.tense)

        # Voice
        if features.voice != "_":
            feats["Voice"] = features.voice

        # Mood and VerbForm
        if features.mood in ["Inf", "Part", "Ger", "Gdv"]:
            feats["VerbForm"] = features.mood
        elif features.mood != "_":
            feats["Mood"] = features.mood
            feats["VerbForm"] = "Fin"

        # Degree
        if features.degree != "_":
            feats["Degree"] = features.degree

        # Definiteness
        if features.definiteness != "_":
            feats["Definite"] = features.definiteness

        return feats

    @classmethod
    def token_to_proiel(cls, token: ConlluToken) -> str:
        """Convert CoNLL-U token directly to PROIEL code."""
        features = cls.conllu_to_features(token)
        return cls.features_to_proiel(features)

    @classmethod
    def get_tense_name(cls, code: str) -> str:
        """Get human-readable tense name from single-letter code."""
        return cls.TENSE_NAMES.get(code, "unknown")

    @classmethod
    def normalize_tense(cls, tense: str) -> str:
        """
        Normalize tense representations to standard single letter codes.
        Handles various input formats (CoNLL-U features, PROIEL codes, full names).
        """
        # Already a single letter code
        if len(tense) == 1 and tense in "pirsualf-":
            return tense

        # CoNLL-U format
        if tense in cls.CONLLU_TENSE:
            return cls.CONLLU_TENSE[tense]

        # Full names
        name_map = {
            "present": "p",
            "imperfect": "i",
            "perfect": "r",
            "pluperfect": "s",
            "aorist": "a",
            "future": "u",
            "future_perfect": "l",
        }
        return name_map.get(tense.lower(), "-")

    @classmethod
    def normalize_voice(cls, voice: str) -> str:
        """Normalize voice to single letter code."""
        if len(voice) == 1 and voice in "ampe-":
            return voice

        voice_map = {
            "Act": "a", "active": "a",
            "Mid": "m", "middle": "m",
            "Pass": "p", "passive": "p",
            "MidPass": "e", "middle-passive": "e", "midpass": "e",
        }
        return voice_map.get(voice, "-")

    @classmethod
    def normalize_mood(cls, mood: str) -> str:
        """Normalize mood to single letter code."""
        if len(mood) == 1 and mood in "ismondpg-":
            return mood

        mood_map = {
            "Ind": "i", "indicative": "i",
            "Sub": "s", "subjunctive": "s",
            "Imp": "m", "imperative": "m",
            "Opt": "o", "optative": "o",
            "Inf": "n", "infinitive": "n",
            "Part": "p", "participle": "p",
        }
        return mood_map.get(mood, "-")

    # PronType normalization mapping
    PRONTYPE_CANONICAL = {
        "Prs": "Prs", "prs": "Prs",
        "Dem": "Dem", "dem": "Dem",
        "Rel": "Rel", "rel": "Rel",
        "Int": "Int", "int": "Int",
        "Ind": "Ind", "ind": "Ind",
        "Rcp": "Rcp", "rcp": "Rcp",
        "Tot": "Tot", "tot": "Tot",
        "Neg": "Neg", "neg": "Neg",
        "Exc": "Exc", "exc": "Exc",
        "_": "_",
    }

    @classmethod
    def normalize_prontype(cls, prontype: str) -> str:
        """Normalize PronType value to canonical form."""
        return cls.PRONTYPE_CANONICAL.get(prontype, prontype)

    @classmethod
    def get_feature_code(
        cls,
        tense: str = "-",
        voice: str = "-",
        mood: str = "-"
    ) -> str:
        """
        Get a 3-letter verbal feature code (tense+voice+mood).

        Args:
            tense: Tense value (any format)
            voice: Voice value (any format)
            mood: Mood value (any format)

        Returns:
            3-letter code like "aai" (aorist active indicative)
        """
        t = cls.normalize_tense(tense)
        v = cls.normalize_voice(voice)
        m = cls.normalize_mood(mood)
        return f"{t}{v}{m}"

    @classmethod
    def parse_feature_code(cls, code: str) -> Tuple[str, str, str]:
        """
        Parse a 3-letter verbal feature code.

        Args:
            code: 3-letter code like "aai"

        Returns:
            Tuple of (tense, voice, mood) in full names
        """
        if len(code) < 3:
            code = code.ljust(3, "-")

        tense = cls.TENSE_NAMES.get(code[0], "unknown")
        voice_names = {"a": "active", "m": "middle", "p": "passive", "e": "middle-passive"}
        mood_names = {
            "i": "indicative", "s": "subjunctive", "m": "imperative",
            "o": "optative", "n": "infinitive", "p": "participle"
        }

        voice = voice_names.get(code[1], "unknown")
        mood = mood_names.get(code[2], "unknown")

        return tense, voice, mood
