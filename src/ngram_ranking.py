import argparse
import json
import os
from typing import Dict, List, Literal
from xml.etree import ElementTree as ET

import kenlm
from utils import SenseInventory, stem_basename_suffix
from utils.wsd import read_wsd_keys


def score_raganato_dataset(xml_data_path: str, txt_gold_keys_path: str, ngram_model_path: str, ngram_size: int,
                           dump_type: Literal["xml", "json", "all"] = "all") -> None:

    valid_dump_types = ["xml", "json", "all"]
    if dump_type not in valid_dump_types:
        raise ValueError(f"`dump_type` must be one among {valid_dump_types}, not '{dump_type}'")

    if not os.path.isfile(xml_data_path):
        raise ValueError(f"{xml_data_path} is not a valid xml data file")

    lemma_keys_dict = read_wsd_keys(txt_gold_keys_path)

    if not os.path.isfile(ngram_model_path):
        raise ValueError(f"{ngram_model_path} is not a valid arpa ngram file")
    
    model = kenlm.LanguageModel(ngram_model_path)

    corpus = ET.parse(xml_data_path)

    samples = dict()

    inventory = SenseInventory()

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
            "bn_esc_predictions": list(),
            "lm_probability": float(sent_xml.attrib.get("lm_probability")),
            "logit_probability": float(sent_xml.attrib.get("logit_probability"))
        }
        samples[sentence_id].append(sample)

        bn_sense_ids: List[str] = list()
        # for each inner xml token (either <instance> or <wf>)
        for token_i, token_xml in enumerate(sent_xml):

            sample["tokens"].append(token_xml.text)

            # consider only instance tokens (ignore all <wf>)
            if token_xml.tag == "instance":

                sample["senses"].append(token_xml.text)
                sample["sense_indices"].append(token_i)

                token_id = token_xml.attrib.get("id")
                lemma_key = lemma_keys_dict.get(token_id, None)
                bn_sense_id = inventory.lemma_key_to_bn_id(lemma_key)

                if lemma_key is not None:

                    bn_sense_ids.append(bn_sense_id)
                    # take into consideration only the last `ngram_size` sense ids
                    bn_sense_ids_window = bn_sense_ids # bn_sense_ids[-ngram_size:]
                    score = model.score(" ".join(bn_sense_ids_window))

                    token_xml.set("wsd_lm_scores", str(score))
                    token_xml.set("esc_prediction", lemma_key)
                    token_xml.set("bn_esc_prediction", bn_sense_id)

                    sample["wsd_lm_scores"].append(score)
                    sample["esc_predictions"].append(lemma_key)
                    sample["bn_esc_predictions"].append(bn_sense_id)

    scored_dataset_path = f"{os.path.dirname(xml_data_path)}/{stem_basename_suffix(xml_data_path)}_ranked"

    if dump_type == "xml" or dump_type == "all":
        ET.indent(corpus, space="", level=0)
        corpus.write(f"{scored_dataset_path}.data.xml", encoding="UTF-8", xml_declaration=True)

    if dump_type == "json" or dump_type == "all":
        with open(f"{scored_dataset_path}.json", "w") as f:
            json.dump(samples, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("--wsd-dataset-path", type=str, required=True)
    parser.add_argument("--wsd-labels-path", type=str, required=True)
    parser.add_argument("--ngram-model-path", type=str, required=True)
    # default + not required
    parser.add_argument("--dump-type", type=str, default="all")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    score_raganato_dataset(xml_data_path=args.wsd_dataset_path, txt_gold_keys_path=args.wsd_labels_path,
                           ngram_model_path=args.ngram_model_path, dump_type=args.dump_type)


if __name__ == "__main__":
    main()
