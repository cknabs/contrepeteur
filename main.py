import functools
import sys
from collections import defaultdict
from itertools import combinations
from typing import List

import numpy as np
import spacy
from spacy.language import Language
from spacy.pipeline import Tagger
from spacy.tokens import Doc, Token

from setup import get_phoneme, get_tags, phoneme2words, SPACY_FILE, setup

DEBUG = False
MAX_PERMUTATION_LENGTH = 3


# Helper
def sub(lst: List, start: int, length=None) -> List:
    if length is None:
        length = len(lst)
    if length > 0:
        return lst[start:start + length]
    else:
        return lst[0:0]


# Searching

Token.set_extension('phoneme', getter=lambda t: get_phoneme(t.text))

Doc.set_extension('tag_scores', default=None)
Token.set_extension('tag_scores', getter=lambda token: token.doc.tag_scores[token.i])


# TODO: use something like https://github.com/explosion/spaCy/issues/5399 to have a tokenizer for strings and lists of strings

# https://github.com/explosion/spaCy/issues/2087
class ProbabilityTagger(Tagger):
    def predict(self, docs):
        tokvecs = self.model.tok2vec(docs)
        scores = self.model.softmax(tokvecs)
        guesses = []
        for i, doc_scores in enumerate(scores):
            docs[i]._.tag_scores = doc_scores
            doc_guesses = doc_scores.argmax(axis=1)

            if not isinstance(doc_guesses, np.ndarray):
                doc_guesses = doc_guesses.get()
            guesses.append(doc_guesses)
        return guesses, tokvecs


Language.factories['tagger'] = lambda nlp, **cfg: ProbabilityTagger(nlp.vocab, **cfg)


def get_permutations(phonemes: List[str]) -> List[str]:
    for (i1, p1), (i2, p2) in combinations(enumerate(phonemes), 2):
        for length1 in range(min(MAX_PERMUTATION_LENGTH, 1 + len(p1))):
            for length2 in range(min(MAX_PERMUTATION_LENGTH, 1 + len(p2))):
                if length1 == 0 and length2 == 0:
                    continue
                for i in range(len(p1) - length1 + 1):
                    for j in range(len(p2) - length2 + 1):
                        p1_prime = sub(p1, 0, i) + sub(p2, j, length2) + sub(p1, i + length2)
                        p2_prime = sub(p2, 0, j) + sub(p1, i, length1) + sub(p2, j + length1)
                        phonemes_prime = phonemes.copy()
                        phonemes_prime[i1] = p1_prime
                        phonemes_prime[i2] = p2_prime
                        yield (i1, p1_prime), (i2, p2_prime)


def get_tag_prob(sentence: str) -> float:
    doc = nlp(sentence)
    scores = doc._.tag_scores
    return scores.max(axis=1).prod()


def get_dep_prob(sentence: str) -> float:
    # Parse sentence using beam search to get probabilities
    with nlp.disable_pipes('parser'):
        beam_doc = nlp(sentence)
    dep_scores = defaultdict(float)
    beams = nlp.parser.beam_parse([beam_doc], beam_width=16, beam_density=0.0001)
    for beam in beams:
        for score, deps in nlp.parser.moves.get_beam_parses(beam):
            for head, elem, label in deps:
                dep_scores[(head, elem, label)] += score
    # Compare probabilities to output
    doc = nlp(sentence)
    scores = np.array([dep_scores[t.head.i, t.i, t.dep_] for t in doc])
    return scores.prod()


def get_prob(sentence: str) -> float:
    return get_tag_prob(sentence) * get_dep_prob(sentence)


def is_valid(old, new, i1: int, i2: int) -> (bool, bool):
    m, n = old[i1], old[i2]
    m_new, n_new = new[i1], new[i2]
    gram_m = get_tags(m).intersection(get_tags(m_new))
    gram_n = get_tags(n).intersection(get_tags(n_new))
    old_tags = [t.tag_ for t in nlp(' '.join(old))]
    new_tags = [t.tag_ for t in nlp(' '.join(new))]
    return len(gram_m) > 0 and len(gram_n) > 0, old_tags == new_tags


def search(sentence: str):
    doc = nlp(sentence)

    phonemes = [t._.phoneme for t in doc]
    words = [t.text for t in doc]

    for (i1, p1_prime), (i2, p2_prime) in get_permutations(phonemes):
        p1, p2 = phonemes[i1], phonemes[i2]

        for m_prime in [x for x in phoneme2words(p1_prime) if get_phoneme(x) != p1]:
            for n_prime in [x for x in phoneme2words(p2_prime) if get_phoneme(x) != p2]:
                new_words = words.copy()
                new_words[i1] = m_prime
                new_words[i2] = n_prime
                b1, b2 = is_valid(words, new_words, i1, i2)
                if b1 or b2:
                    new_sentence = ' '.join(new_words)
                    yield new_sentence, get_prob(new_sentence), b1, b2


if __name__ == '__main__':
    nlp = spacy.load(SPACY_FILE)
    print(f"Loaded SpaCy language package {SPACY_FILE}")
    setup()
    for sentence in sys.argv[1:]:
        print(sentence)

        out = list(set(search(sentence)))  # Remove duplicates


        def cmp(tup1, tup2) -> int:
            n1 = [tup1[2], tup1[3]].count(True)
            n2 = [tup2[2], tup2[3]].count(True)
            if n1 != n2:
                return n2 - n1
            else:
                prob1, prob2 = tup1[1], tup2[1]
                return prob2 - prob1


        out.sort(key=functools.cmp_to_key(cmp))
        for o in out:
            print(f"{o[0]}\t\t{o[1]}\t{'L' if o[2] else ' '}{'S' if o[3] else ' '}")
