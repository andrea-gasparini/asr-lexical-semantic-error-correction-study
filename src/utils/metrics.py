import math
from collections import Counter
from functools import cache

from tqdm import tqdm

from utils.os import get_num_lines


class PointwiseMutualInformation:

    def __init__(self, source_file_path: str) -> None:
        self.source_file_path = source_file_path
        self.unigram_frequences, self.bigram_frequences = Counter(), Counter()
        self.__compute_frequences()

    def __call__(self, token1: str, token2: str) -> float:
        return self.pmi(token1, token2)

    def __compute_frequences(self) -> None:
        with open(self.source_file_path) as source_file:
            for line in tqdm(source_file, total=get_num_lines(self.source_file_path)):
                tokens = line.split(" ")
                for i, token in enumerate(tokens):
                    self.unigram_frequences[token] += 1
                    if len(tokens) > i + 1:
                        self.bigram_frequences[f"{token} {tokens[i + 1]}"] += 1

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
