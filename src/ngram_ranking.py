import os
from typing import Dict, List
from xml.etree import ElementTree as ET
import kenlm


def read_wsd_gold_keys(txt_path: str) -> Dict[str, str]:
    """
    Reads the gold keys of a WSD corpus from a txt file
    and parses it into a dictionary that goes from tokens ids to wordnet sense ids.

    Args:
        txt_path: gold keys file

    Returns: tokens ids to wordnet sense ids dictionary
    """
    if not os.path.isfile(txt_path):
        raise Exception(f"{txt_path} is not a valid txt gold keys file")

    with open(txt_path) as f:
        gold_keys = [line.strip().split(" ") for line in f]

    sense_ids_dict = dict()
    for gold_key in gold_keys:
        token_id = gold_key[0]
        sense_id = gold_key[1]  # ignore eventual secondary senses ([2:])
        sense_ids_dict[token_id] = sense_id

    return sense_ids_dict


def score_raganato_dataset(xml_data_path: str, txt_gold_keys_path: str,
                           ngram_model_path: str, ngram_size: int,
                           output_file_path: str) -> None:

    if not os.path.isfile(xml_data_path):
        raise Exception(f"{xml_data_path} is not a valid xml data file")

    sense_ids_dict = read_wsd_gold_keys(txt_gold_keys_path)

    if not os.path.isfile(ngram_model_path):
        raise Exception(f"{ngram_model_path} is not a valid arpa ngram file")
    
    model = kenlm.LanguageModel(ngram_model_path)

    corpus = ET.parse(xml_data_path)

    # iterate over <sentence> tags from the given xml file
    for sent_i, sent_xml in enumerate(corpus.iter("sentence")):

        sense_ids: List[str] = list()
        # for each inner xml token (either <instance> or <wf>)
        for token_i, token_xml in enumerate(sent_xml):

            # consider only instance tokens (ignore all <wf>)
            if token_xml.tag == "instance":

                token_id = token_xml.attrib.get("id")
                sense_id = sense_ids_dict.get(token_id, None)

                if sense_id is not None:

                    sense_ids.append(sense_id)
                    # take into consideration only the last `ngram_size` sense ids
                    sense_ids_window = sense_ids[-ngram_size:]
                    score = model.score(" ".join(sense_ids_window))

                    token_xml.set("esc_prediction", sense_id)
                    token_xml.set("wsd_lm_score", str(score))

                else:

                    print(f"Token {token_id} does not have a prediction in {txt_gold_keys_path}")

    ET.indent(corpus, space="", level=0)
    corpus.write(output_file_path, encoding="UTF-8", xml_declaration=True)

                    
if __name__ == "__main__":

    score_raganato_dataset("../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml",
                           "../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt",
                           "../models/ngrams/4gram.arpa", 4,
                           "../data/new.data.xml")
