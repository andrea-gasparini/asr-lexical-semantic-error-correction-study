import json
import math
from collections import Counter
from functools import cache
from typing import Dict, Optional

from tqdm import tqdm

from utils.os import get_num_lines


class PointwiseMutualInformation:
    # serialization constants
    # json attrs will get serialized into a single json file
    JSON_ATTRS = ("source_file_path", "unigram_frequences", "bigram_frequences")

    def __init__(self, source_file_path: str,
                 unigram_frequences: Optional[Dict] = None,
                 bigram_frequences: Optional[Dict] = None) -> None:

        self.source_file_path = source_file_path

        if unigram_frequences is None or bigram_frequences is None:
            self.unigram_frequences = Counter()
            self.bigram_frequences = Counter()
            self.__compute_frequences()
        else:
            self.unigram_frequences = unigram_frequences
            self.bigram_frequences = bigram_frequences

    def __call__(self, token1: str, token2: str) -> float:
        return self.pmi(token1, token2)

    def __compute_frequences(self) -> None:
        """Computes unigram and bigram frequences from the source file."""
        with open(self.source_file_path) as source_file:
            for line in tqdm(source_file, total=get_num_lines(self.source_file_path)):
                tokens = line.split(" ")
                for i, token in enumerate(tokens):
                    self.unigram_frequences[token] += 1
                    if len(tokens) > i + 1:
                        self.bigram_frequences[f"{token} {tokens[i + 1]}"] += 1

    @property
    def __serializable_attrs(self) -> Dict:
        """Gets a dictionary of the attributes to serialize to json."""
        json_attrs = dict()
        for attr in self.JSON_ATTRS:
            val = getattr(self, attr)
            if val is None:
                raise ValueError(f"attribute {attr} not found. Cannot serialize")
            json_attrs[attr] = val
        return json_attrs

    @cache
    def pmi(self, token1: str, token2: str) -> float:
        if token1 not in self.unigram_frequences or token2 not in self.unigram_frequences:
            return 0
        prob_token1 = self.unigram_frequences[token1] / float(sum(self.unigram_frequences.values()))
        prob_token2 = self.unigram_frequences[token2] / float(sum(self.unigram_frequences.values()))
        prob_token1_token2 = self.bigram_frequences[f"{token1} {token2}"] / float(sum(self.bigram_frequences.values()))
        return math.log(prob_token1_token2 / float(prob_token1 * prob_token2), 2)

    def ppmi(self, token1: str, token2: str) -> float:
        """
        The positive pointwise mutual information (PPMI) measure is defined by setting negative values of PMI to zero.
        """
        return max((self.pmi(token1, token2), 0))

    def npmi(self, token1: str, token2: str) -> float:
        """
        Pointwise mutual information can be normalized (NPMI) between [-1,+1] resulting in -1 (in the limit) for never
        occurring together, 0 for independence, and +1 for complete co-occurrence.
        """
        if f"{token1} {token2}" not in self.bigram_frequences:
            return -1
        prob_token1_token2 = self.bigram_frequences[f"{token1} {token2}"] / float(sum(self.bigram_frequences.values()))
        return self.pmi(token1, token2) / -math.log(prob_token1_token2, 2)

    def save_to_dir(self, filepath: str) -> None:
        """Saves to a directory."""
        with open(filepath, "w") as f:
            json.dump(self.__serializable_attrs, f)

    @classmethod
    def load_from_dir(cls, filepath: str) -> "PointwiseMutualInformation":
        """Loads from a directory."""
        with open(filepath) as f:
            json_attrs = json.load(f)

        if set(json_attrs.keys()) != set(cls.JSON_ATTRS):
            raise ValueError(f"Expected serialized attributes to be {cls.JSON_ATTRS} but found {json_attrs.keys()}")

        return cls(**json_attrs)
