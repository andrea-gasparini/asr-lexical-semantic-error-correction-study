import argparse
import json
import os
from typing import Dict, List, Literal
from xml.etree import ElementTree as ET
import kenlm

from constants import *
from utils import stem_basename_suffix, get_basename


def read_wsd_gold_keys(txt_path: str) -> Dict[str, str]:
    """
    Reads the gold keys of a WSD corpus from a txt file
    and parses it into a dictionary that goes from tokens ids to wordnet sense ids.

    Args:
        txt_path: gold keys file

    Returns: tokens ids to wordnet sense ids dictionary
    """
    if not os.path.isfile(txt_path):
        raise ValueError(f"{txt_path} is not a valid txt gold keys file")

    with open(txt_path) as f:
        gold_keys = [line.strip().split(" ") for line in f]

    sense_ids_dict = dict()
    for gold_key in gold_keys:
        if len(gold_key) > 1:
            token_id = gold_key[0]
            sense_id = gold_key[1]  # ignore eventual secondary senses ([2:])
            sense_ids_dict[token_id] = sense_id
        else:
            # TODO: implement logger
            # print(f"Token {token_id} does not have a prediction in {get_basename(txt_path)}")
            pass

    return sense_ids_dict


def score_raganato_dataset(xml_data_path: str, txt_gold_keys_path: str, ngram_model_path: str, ngram_size: int,
                           dump_type: Literal["xml", "json", "all"] = "all") -> None:

    valid_dump_types = ["xml", "json", "all"]
    if dump_type not in valid_dump_types:
        raise ValueError(f"`dump_type` must be one among {valid_dump_types}, not '{dump_type}'")

    if not os.path.isfile(xml_data_path):
        raise ValueError(f"{xml_data_path} is not a valid xml data file")

    sense_ids_dict = read_wsd_gold_keys(txt_gold_keys_path)

    if not os.path.isfile(ngram_model_path):
        raise ValueError(f"{ngram_model_path} is not a valid arpa ngram file")
    
    model = kenlm.LanguageModel(ngram_model_path)

    corpus = ET.parse(xml_data_path)

    samples = dict()

    # iterate over <sentence> tags from the given xml file
    for sent_i, sent_xml in enumerate(corpus.iter("sentence")):

        sentence_id = sent_xml.attrib.get("sentence_id")
        if sentence_id not in samples:
            samples[sentence_id] = list()

        sample = {
            "tokens": list(),
            "senses": list(),
            "sense_indices": list(),
            "wsd_lm_scores": list(),
            "esc_predictions": list(),
            "lm_probability": float(sent_xml.attrib.get("lm_probability")),
            "logit_probability": float(sent_xml.attrib.get("logit_probability"))
        }
        samples[sentence_id].append(sample)

        sense_ids: List[str] = list()
        # for each inner xml token (either <instance> or <wf>)
        for token_i, token_xml in enumerate(sent_xml):

            sample["tokens"].append(token_xml.text)

            # consider only instance tokens (ignore all <wf>)
            if token_xml.tag == "instance":

                sample["senses"].append(token_xml.text)
                sample["sense_indices"].append(token_i)

                token_id = token_xml.attrib.get("id")
                sense_id = sense_ids_dict.get(token_id, None)

                if sense_id is not None:

                    sense_ids.append(sense_id)
                    # take into consideration only the last `ngram_size` sense ids
                    sense_ids_window = sense_ids[-ngram_size:]
                    score = model.score(" ".join(sense_ids_window))

                    token_xml.set("esc_prediction", sense_id)
                    token_xml.set("wsd_lm_score", str(score))

                    sample["wsd_lm_scores"].append(score)
                    sample["esc_predictions"].append(sense_id)

                else:

                    sample["wsd_lm_scores"].append(None)
                    sample["esc_predictions"].append(None)

    scored_dataset_path = f"{os.path.dirname(xml_data_path)}/{stem_basename_suffix(xml_data_path)}_ranked"

    if dump_type == "xml" or dump_type == "all":
        ET.indent(corpus, space="", level=0)
        corpus.write(f"{scored_dataset_path}.data.xml", encoding="UTF-8", xml_declaration=True)

    if dump_type == "json" or dump_type == "all":
        with open(f"{scored_dataset_path}.json", "w") as f:
            json.dump(samples, f, indent=2)

                    
if __name__ == "__main__":

    score_raganato_dataset("../data/librispeech/librispeech_test_all.data.xml",
                           "../data/librispeech/librispeech_test_all.silver.key.txt",
                           "../models/ngrams/4gram.binary", 4)
