from functools import cache
import os
from pathlib import Path
from typing import List, Optional, Dict
from xml.etree import ElementTree as ET
import nltk
from nltk.corpus import wordnet
from nltk.corpus.reader import Synset

from constants import LEMMA2IDS_FILE_PATH, WN2BN_IDS_FILE_PATH

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
    Efficiently pre-computes the total number of lines contained in an ascii file.

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


def dict_to_list(dict_of_lists: Dict[str, List]) -> List[Dict]:
    return [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]


def list_to_dict(list_of_dicts: List[Dict]) -> Dict[str, List]:
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}


# TODO: move in another module
# TODO: implement Singleton pattern
class SenseInventory:

    def __init__(self, lemma2ids_file_path: str = LEMMA2IDS_FILE_PATH,
                 wn2bn_ids_file_path: str = WN2BN_IDS_FILE_PATH) -> None:

        self.__lemma_keys_to_wn_ids = dict()

        with open(lemma2ids_file_path) as f:
            for line in f:
                lemma_key, wn_pos_offset = line.split()
                self.__lemma_keys_to_wn_ids[lemma_key] = f"wn:{wn_pos_offset}"

        self.__wn_to_bn_ids = dict()

        with open(wn2bn_ids_file_path) as f:
            for line in f:
                wn_id, bn_id = line.split()
                self.__wn_to_bn_ids[wn_id] = bn_id

    def lemma_key_to_wn_id(self, lemma_key: str) -> Optional[str]:
        return self.__lemma_keys_to_wn_ids.get(lemma_key, None)

    def lemma_key_to_bn_id(self, lemma_key: str) -> Optional[str]:
        return self.wn_id_to_bn(self.lemma_key_to_wn_id(lemma_key))

    def wn_id_to_bn(self, wn_id: str) -> Optional[str]:
        return self.__wn_to_bn_ids.get(wn_id, None)
