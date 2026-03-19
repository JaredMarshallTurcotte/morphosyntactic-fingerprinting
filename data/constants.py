"""
Core constants for the MorphTagTransformer model.

PROIEL morphological code vocabularies, dependency relation mappings,
and sequence length settings extracted from the full HMSM dataset module.
"""

# PROIEL 10-position morphology code vocabularies.
# Each position has a distinct character set; per-position one-hot encoding
# avoids the collisions inherent in ord(char) % 10.
PROIEL_POSITION_VOCABS = [
    list('NVADCRPMIGX-'),   # pos 0: POS (12)
    list('123-'),            # pos 1: person (4)
    list('spd-'),            # pos 2: number (4)
    list('pirsaulf-'),       # pos 3: tense (9)
    list('ismonpdg-'),       # pos 4: mood (9)
    list('ampe-'),           # pos 5: voice (5, incl. e=mid-pass)
    list('mfnc-'),           # pos 6: gender (5, incl. c=common)
    list('nagdvo-'),         # pos 7: case (7, incl. o=oblique)
    list('pcs-'),            # pos 8: degree (4)
    list('ws-'),             # pos 9: strength/definiteness (3)
]
MORPH_CODE_DIM = sum(len(v) for v in PROIEL_POSITION_VOCABS)  # = 62

MAX_MORPH_SEQ_LEN = 64

DEPREL_VOCAB = [
    'root', 'nsubj', 'obj', 'iobj', 'obl', 'advmod', 'amod', 'nmod',
    'conj', 'cc', 'mark', 'advcl', 'acl', 'xcomp', 'ccomp', 'aux',
    'cop', 'det', 'case', 'punct', 'other',
]
DEPREL_TO_IDX = {dr: i for i, dr in enumerate(DEPREL_VOCAB)}
DEPREL_PAD_IDX = len(DEPREL_VOCAB)  # 21
