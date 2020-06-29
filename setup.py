import csv
from collections import defaultdict
from itertools import product
from typing import Set

LEXIQUE_FILE = 'Lexique/Lexique383.tsv'
SPACY_FILE = 'fr_core_news_sm'

word_phoneme = defaultdict(str)
phoneme_words = defaultdict(set)
word_tags = defaultdict(set)
word_freq = defaultdict(float)


# Getters
def get_word_freq(word, info):
    return word_freq[word + ':' + info]


def get_phoneme(word: str) -> str:
    return word_phoneme[word.lower()]


def phoneme2words(phoneme: str) -> Set[str]:
    return phoneme_words[phoneme]


def get_tags(word: str) -> Set[str]:
    return word_tags[word]


# Helpers for setup
def read_tags(row: dict):
    gram = row['cgram']
    if gram == 'VER':
        infoverbe = row['infover']
        return ['VER_' + iv for iv in infoverbe.split(';')[:-1]]
    else:
        genre = row['genre']
        nombre = row['nombre']
        genres = ['m', 'f'] if genre == '' else [genre]
        nombres = ['s', 'p'] if nombre == '' else [nombre]
        return [gram + '_' + g + n for g, n in product(genres, nombres)]


def read_freq(row):
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
            freq = read_freq(row)

            word_phoneme[word] = phon
            if phon not in phoneme_words:
                phoneme_words[phon] = set()
            phoneme_words[phon].add(word)

            if word not in word_tags:
                word_tags[word] = set()
            word_tags[word].update(read_tags(row))

            for o in read_tags(row):
                word_freq[word + ':' + o] = freq
    print(f"{len(word_phoneme)} orthographes chargées")
    print("=" * 50)
