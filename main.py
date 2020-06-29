import sys
from collections import defaultdict
from itertools import combinations
from typing import List

import numpy as np
import spacy
from spacy.language import Language
from spacy.pipeline import Tagger
from spacy.tokens import Doc, Token

from setup import *


# Helper
def sub(lst: List, start: int, length=None) -> List:
    if length is None:
        length = len(lst)
    if length > 0:
        return lst[start:start + length]
    else:
        return lst[0:0]


DEBUG = False


def is_match(m, m_prime, cat_m):
    if DEBUG:
        print(f"\t{m} ~> {m_prime}\t{cat_m}")
    return True


# Searching

# Doc.set_extension('phonemes', default=None)
Token.set_extension('phoneme', getter=lambda t: get_phoneme(t.text))

Doc.set_extension('tag_scores', default=None)
Token.set_extension('tag_scores', getter=lambda token: token.doc.tag_scores[token.i])


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
        min_len = min(len(p1), len(p2))
        for length in range(1, min(3, min_len)):
            for i in range(len(p1) - length + 1):
                for j in range(len(p2) - length + 1):
                    x = sub(p1, i, length)
                    y = sub(p2, j, length)

                    p1_prime = sub(p1, 0, i) + y + sub(p1, i + length)
                    p2_prime = sub(p2, 0, j) + x + sub(p2, j + length)
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


def get_prob(sentence: str):
    return get_tag_prob(sentence) * get_dep_prob(sentence)


def search(sentence):
    doc = nlp(sentence)
    # for token in doc:
    #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #     token.shape_, token.is_alpha, token.is_stop, token._.phoneme)

    # phonemes = doc._.phonemes
    phonemes = [t._.phoneme for t in doc]
    words = [t.text for t in doc]

    for (i1, p1_prime), (i2, p2_prime) in get_permutations(phonemes):
        if p1_prime not in INV or p2_prime not in INV:
            continue

        p1 = phonemes[i1]
        p2 = phonemes[i2]
        m = words[i1]
        n = words[i2]

        for m_prime in [x for x in INV[p1_prime] if LEXIQUE[x] != p1]:
            for n_prime in [x for x in INV[p2_prime] if LEXIQUE[x] != p2]:

                gram_m = G[m].intersection(G[m_prime])
                gram_n = G[n].intersection(G[n_prime])
                for cat_m in gram_m:
                    for cat_n in gram_n:
                        c_m = is_match(m, m_prime, cat_m)
                        c_n = is_match(n, n_prime, cat_n)
                        if c_m and c_n:
                            new_words = words.copy()
                            new_words[words.index(m)] = m_prime
                            new_words[words.index(n)] = n_prime
                            new_sentence = ' '.join(new_words)
                            yield new_sentence, get_prob(new_sentence)
                        else:
                            pass


if __name__ == '__main__':
    nlp = spacy.load(SPACY_FILE)
    print(f"Loaded language package ({len(nlp.vocab)} words)")
    setup()
    for sentence in sys.argv[1:]:
        print(sentence)

        out = list(set(search(sentence)))  # Remove duplicates
        out.sort(key=lambda t: t[1], reverse=True)
        for o in out:
            print(f"{o[0]}\t\t{o[1]}")
