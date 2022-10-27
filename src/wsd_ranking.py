import argparse
import json
import os
from typing import Dict, List, Literal
from xml.etree import ElementTree as ET
from tqdm import tqdm

import kenlm
from utils import SenseInventory, stem_basename_suffix
from utils.metrics import PointwiseMutualInformation
from utils.wsd import read_wsd_keys


VALID_DUMP_TYPES = ["xml", "json", "all"]


def tag_wsd_predictions(xml_data_path: str, txt_gold_keys_path: str, ngram_model_path: str, pmi_attrs_path: str,
                        dump_type: Literal["xml", "json", "all"] = "all") -> None:
    if dump_type not in VALID_DUMP_TYPES:
        raise ValueError(f"`dump_type` must be one among {VALID_DUMP_TYPES}, but is '{dump_type}'")

    if not os.path.isfile(xml_data_path):
        raise ValueError(f"{xml_data_path} is not a valid xml data file")

    lemma_keys_dict = read_wsd_keys(txt_gold_keys_path)

    if not os.path.isfile(ngram_model_path):
        raise ValueError(f"{ngram_model_path} is not a valid arpa ngram file")
    
    if not os.path.isfile(pmi_attrs_path):
        raise ValueError(f"{pmi_attrs_path} is not a valid json pmi's dump file")

    language_model = kenlm.LanguageModel(ngram_model_path)
    pmi_model = PointwiseMutualInformation.load_from_dir(pmi_attrs_path)

    corpus = ET.parse(xml_data_path)

    samples = dict()

    inventory = SenseInventory()

    # iterate over <sentence> tags from the given xml file
    for sent_xml in tqdm(list(corpus.iter("sentence"))):

        sentence_id = sent_xml.attrib.get("sentence_id")
        if sentence_id not in samples:
            samples[sentence_id] = list()

        sample = {
            "transcription": sent_xml.attrib.get("transcription"),
            "tokens": list(),
            "senses": list(),
            "sense_indices": list(),
            "wsd_pmi_scores": list(),
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

                token_id = token_xml.attrib.get("id")
                lemma_key = lemma_keys_dict.get(token_id, None)
                bn_sense_id = inventory.lemma_key_to_bn_id(lemma_key)

                if lemma_key is not None:
                    bn_sense_ids.append(bn_sense_id)
                    # take into consideration only the last `ngram_size` sense ids
                    # bn_sense_ids_window = bn_sense_ids[-ngram_size:]
                    lm_score = language_model.score(" ".join(bn_sense_ids))

                    token_xml.set("wsd_lm_scores", str(lm_score))
                    token_xml.set("esc_prediction", lemma_key)
                    token_xml.set("bn_esc_prediction", bn_sense_id)

                    sample["wsd_lm_scores"].append(lm_score)
                    sample["esc_predictions"].append(lemma_key)
                    sample["bn_esc_predictions"].append(bn_sense_id)
                    sample["senses"].append(token_xml.text)
                    sample["sense_indices"].append(token_i)

        for i in range(len(bn_sense_ids)):
            pmi_score = pmi_model.compute_average_pmi(i, bn_sense_ids)
            sample["wsd_pmi_scores"].append(pmi_score)

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
    parser.add_argument("--pmi-attrs-path", type=str, required=True)
    # default + not required
    parser.add_argument("--dump-type", type=str, default="all", choices=VALID_DUMP_TYPES)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tag_wsd_predictions(xml_data_path=args.wsd_dataset_path, txt_gold_keys_path=args.wsd_labels_path,
                        ngram_model_path=args.ngram_model_path, pmi_attrs_path=args.pmi_attrs_path,
                        dump_type=args.dump_type)


if __name__ == "__main__":
    main()
