import os
from functools import cache
from typing import List, Optional, Dict

import nltk
from nltk.corpus import wordnet
from nltk.corpus.reader import Synset

from constants import LEMMA2IDS_FILE_PATH, WN2BN_IDS_FILE_PATH

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


def download_wordnet_resources() -> None:
    nltk.download("wordnet")


@cache
def synsets_from_lemmapos(lemma: str, pos: str) -> List[Synset]:
    return wordnet.synsets(lemma, pos)


def read_wsd_keys(txt_keys_path: str) -> Dict[str, str]:
    """
    Reads the keys of a WSD corpus from a txt file (following Raganato's format)
    and parses it into a dictionary that goes from tokens ids to wordnet lemma keys.

    Args:
        txt_keys_path (`str`):
            Path to a txt labels keys file.

    Returns:
        `Dict[str, str]`:
            Dictionary from token ids to wordnet lemma keys.
    """
    if not os.path.isfile(txt_keys_path):
        raise ValueError(f"{txt_keys_path} is not a valid txt keys file")

    with open(txt_keys_path) as f:
        lines = [line.strip().split(" ") for line in f]

    lemma_keys_dict = dict()
    for line in lines:
        if len(line) > 1:
            token_id = line[0]
            lemma_key = line[1]  # ignore eventual secondary senses ([2:])
            lemma_keys_dict[token_id] = lemma_key
        else:
            # TODO: implement logger
            # print(f"Token {token_id} does not have a prediction in {get_basename(txt_keys_path)}")
            pass

    return lemma_keys_dict


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
