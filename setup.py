import csv
from itertools import product

LEXIQUE_FILE = 'Lexique/Lexique383.tsv'
SPACY_FILE = 'fr_core_news_sm'

LEXIQUE = {}
INV = {}
G = {}
WORD_FREQ = {}


# Getters
def get_word_freq(word, info):
    return WORD_FREQ[word + ':' + info]


def get_phoneme(word):
    return LEXIQUE[word] if word in LEXIQUE else ''


def phoneme2word(phoneme):
    return INV[phoneme]


# Helpers for setup

def get_tags(row):
    gram = row['cgram']
    if gram == 'VER':
        infoverbe = row['infover']
        return ['VER_' + iv for iv in infoverbe.split(';')]
    else:
        genre = row['genre']
        nombre = row['nombre']
        genres = ['m', 'f'] if genre == '' else [genre]
        nombres = ['s', 'p'] if nombre == '' else [nombre]
        return [gram + '_' + g + n for g, n in product(genres, nombres)]


def get_freq(row):
    f1 = float(row['freqlemfilms2']) / 1000000
    f2 = float(row['freqlemlivres']) / 1000000
    return 0.5 * (f1 + f2)


# Setup
def setup(lexique_file=LEXIQUE_FILE):
    with open(lexique_file, 'r') as lex_in:
        tsv_in = csv.DictReader(lex_in, delimiter='\t')
        for row in tsv_in:
            word = row['ortho']
            phon = row['phon']
            freq = get_freq(row)

            LEXIQUE[word] = phon
            if phon not in INV:
                INV[phon] = set()
            INV[phon].add(word)

            if word not in G:
                G[word] = set()
            G[word].update(get_tags(row))

            for o in get_tags(row):
                WORD_FREQ[word + ':' + o] = freq
    print(f"{len(LEXIQUE)} orthographes chargées")
    print("=" * 50)
