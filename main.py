import csv
import sys
from itertools import combinations, product

import spacy


# Lexique = namedtuple('Lexique', 'ortho phono lemme cgram genre nombre freqlemfilms freqlemlivres freqfilms freqlivres infover nbhomogr nbhomoph islem nblettres nbphon cvcv p-cvcv voisorth voisphon puorth puphon syll nbsyll cv-cv ortheenv phonrev orthosyll')

def get_objs(row):
    word = row[0]
    gram = row[3]
    if gram == 'VER':
        infoverbe = row[10]
        return ['VER_' + iv for iv in infoverbe.split(';')]
    else:
        genre = row[4]
        nombre = row[5]
        if genre == '':
            genres = ['m', 'f']
        else:
            genres = [genre]
        if nombre == '':
            nombres = ['s', 'p']
        else:
            nombres = [nombre]
        return [gram + '_' + g + n for g, n in product(genres, nombres)]


lexique_file = 'Lexique/Lexique383.tsv'

lexique = {}
inv = {}
genre = {}
nombre = {}
infoverbe = {}
G = {}
GG = {}
with open(lexique_file, 'r') as lex_in:
    tsv_in = csv.reader(lex_in, delimiter='\t')

    next(tsv_in)  # Skip header
    for row in tsv_in:
        word = row[0]
        phon = row[1]
        gram = row[3]

        lexique[word] = phon

        genre[word] = row[4]
        nombre[word] = row[5]
        infoverbe[word] = row[10]

        if phon not in inv:
            inv[phon] = set()
        inv[phon].add(word)

        if word not in G:
            G[word] = set()
            GG[word] = set()
        G[word].add(gram)
        GG[word].update(get_objs(row))

print(f"{len(lexique)} orthographes chargées")
nlp = spacy.load('fr_core_news_sm')
print(f"Loaded language package ({len(nlp.vocab)} words)")
print("=" * 50)


# Helper
def sub(l, start, length=None):
    if length is None:
        length = len(l)
    if length > 0:
        return l[start:start + length]
    else:
        return l[0:0]


def cmpstring(x, y):
    if x == y:
        return '=='
    else:
        return '!='


DEBUG = True


def is_match(m, m_prime, cat_m):
    if DEBUG:
        print(f"\t{m} ~> {m_prime}\t{cat_m}")
    return True


def get_phoneme(word):
    if word in lexique:
        return lexique[word]
    else:
        return ''


def phoneme2word(phoneme):
    return inv[phoneme]


# Searching
from spacy.tokens import Token

Token.set_extension('phoneme', getter=lambda t: get_phoneme(t.text))


def search(contrepeterie):
    doc = nlp(contrepeterie)
    # for token in doc:
    #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #     token.shape_, token.is_alpha, token.is_stop, token._.phoneme)

    phonemes = [t._.phoneme for t in doc]
    words = [t.text for t in doc]

    for t1, t2 in combinations(doc, 2):
        p1, p2 = t1._.phoneme, t2._.phoneme
        min_len = min(len(p1), len(p2))
        for l in range(1, min(3, min_len)):
            for i in range(len(p1) - l + 1):
                for j in range(len(p2) - l + 1):
                    # print(sub(w1, i, l), ' <-> ', sub(w2, j, l))

                    x = sub(p1, i, l)
                    y = sub(p2, j, l)

                    w1_prime = sub(p1, 0, i) + y + sub(p1, i + l)
                    w2_prime = sub(p2, 0, j) + x + sub(p2, j + l)

                    # print(i, w1_prime)
                    # print(j, w2_prime)

                    if w1_prime not in inv or w2_prime not in inv:
                        continue

                    m = doc[phonemes.index(p1)].text
                    n = doc[phonemes.index(p2)].text

                    for m_prime in [x for x in inv[w1_prime] if lexique[x] != p1]:
                        for n_prime in [x for x in inv[w2_prime] if lexique[x] != p2]:

                            gram_m = GG[m].intersection(GG[m_prime])
                            gram_n = GG[n].intersection(GG[n_prime])
                            for cat_m in gram_m:
                                for cat_n in gram_n:
                                    print('-' * 30)
                                    c_m = is_match(m, m_prime, cat_m)
                                    c_n = is_match(n, n_prime, cat_n)
                                    if c_m and c_n:
                                        new_words = words.copy()
                                        new_words[words.index(m)] = m_prime
                                        new_words[words.index(n)] = n_prime
                                        print("--MATCH--", ' '.join(new_words))
                                    else:
                                        # print("--MISMATCH--")
                                        pass


for contrepeterie in sys.argv[1:]:
    print(contrepeterie)
    search(contrepeterie)
