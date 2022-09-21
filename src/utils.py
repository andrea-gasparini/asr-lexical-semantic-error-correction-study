from functools import cache
from typing import List
from xml.etree import ElementTree as ET
import nltk
from nltk.corpus import wordnet
from nltk.corpus.reader import Synset

nltk.download("wordnet")


pos_map = {
    # U-POS
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
    "PROPN": "n",
    # PEN
    "AFX": "a",
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "MD": "v",
    "NN": "n",
    "NNP": "n",
    "NNPS": "n",
    "NNS": "n",
    "RB": "r",
    "RP": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
    "WRB": "r",
}


@cache
def synsets_from_lemmapos(lemma: str, pos: str) -> List[Synset]:
    return wordnet.synsets(lemma, pos)
