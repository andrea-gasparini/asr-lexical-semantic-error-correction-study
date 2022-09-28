from functools import cache
import os
from pathlib import Path
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


def get_num_lines(file_name: str) -> int:
    """
    Efficiently computes the total number of lines contained in an ascii file.

    Source:
        https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python/68385697#68385697
    """

    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(file_name, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))

    return count


def stem_basename_suffix(path: str) -> str:

    stemmed_path = Path(path).stem

    if path == stemmed_path:
        return path

    return stem_basename_suffix(stemmed_path)


def get_basename(path: str) -> str:
    return os.path.basename(os.path.normpath(path))
