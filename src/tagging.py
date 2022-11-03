import argparse
import json
import os
from typing import Literal, List
from xml.etree import ElementTree as ET

import kenlm
import datasets
import stanza
from tqdm import tqdm

from constants import *
from models import Wav2Vec2WithLM
from utils.metrics import PointwiseMutualInformation
from utils.wsd import SenseInventory, synsets_from_lemmapos, pos_map, read_wsd_keys
from utils.os import stem_basename_suffix


PREDICTIONS_PATH = f"{DATA_DIR}predictions/"


def generate_librispeech_predictions(hf_model_name: str, predictions_path: str = PREDICTIONS_PATH) -> None:
    w2v2 = Wav2Vec2WithLM.from_pretrained(hf_model_name, f"{NGRAM_PATH}librispeech/4-gram")
    ls_test_other = datasets.Dataset.load_from_disk(f"{DATA_DIR}librispeech/librispeech_test_other")
    ls_test_clean = datasets.Dataset.load_from_disk(f"{DATA_DIR}librispeech/librispeech_test_clean")
    
    model_name = hf_model_name.split("/")[1]
    for dataset, dataset_name in [(ls_test_other, "test_other"), (ls_test_clean, "test_clean")]:       
        predictions = dataset.map(w2v2.beam_search, remove_columns=["file", "audio"])
        predictions.save_to_disk(os.path.join(predictions_path, f"{model_name}-4-gram-{dataset_name}"))


def compute_wsd_dataset_librispeech(model_name: str, predictions_path: str = PREDICTIONS_PATH) -> None:
    model_predictions_path = os.path.join(predictions_path, model_name)
    preds_test_other = datasets.Dataset.load_from_disk(f"{model_predictions_path}-test_other")
    preds_test_clean = datasets.Dataset.load_from_disk(f"{model_predictions_path}-test_clean")
    preds_test_all = datasets.concatenate_datasets([preds_test_clean, preds_test_other]).sort(column="chapter_id")
    compute_wsd_dataset(preds_test_all, "librispeech_test_all", model_name, predictions_path)


def compute_wsd_dataset(dataset: datasets.Dataset, dataset_name: str, model_name: str, predictions_path: str = PREDICTIONS_PATH) -> None:
    """
    Preprocesses the given dataset, containing the predictions of a Wav2Vec2 + LM model, with lemmas and part-of-speech (POS)
    and saves it to a xml WSD dataset file following Raganato's format.    
    
    The WSD dataset format has been introduced by Raganato et al. (2017) in
    [Word Sense Disambiguation: A Unified Evaluation Framework and Empirical Comparison](https://www.aclweb.org/anthology/E17-1010/).
    
    Args:
        dataset (`datasets.Dataset`):
            Dataset containing the predictions of a Wav2Vec2 + LM model.
        dataset_name (`str`):
            Name of the given dataset.
        model_name (`str`):
            Name of the model used to compute the predictions.
        predictions_path (`str`, optional, defaults to `PREDICTIONS_PATH`):
            Path to the directory in which to save the tagged WSD dataset.
    """
    tagging_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_no_ssplit=True)

    assert "candidates" in dataset[0], f'Predictions of model "{model_name}" are not valid ' \
                                       f'since they do not contain a "candidates" key. ' \
                                       f'Make sure they have been generated correctly by a beam search.'

    corpus = ET.Element("corpus", attrib={"lang": "en", "source": f"{dataset_name} + {model_name}"})

    pbar = tqdm(total=len(dataset))
    pbar.set_description("Processing %s" % dataset_name)

    text_cnt, sentence_cnt = 0, 0
    text_id = f"d{text_cnt:03d}"
    chapter_id = dataset[0]["chapter_id"]
    text = ET.SubElement(corpus, "text", attrib={"id": text_id, "chapter_id": str(chapter_id)})

    for sample in dataset:

        if sample["chapter_id"] != chapter_id:
            chapter_id = sample["chapter_id"]
            sentence_cnt = 0
            text_cnt += 1
            text_id = f"d{text_cnt:03d}"
            text = ET.SubElement(corpus, "text", attrib={"id": text_id, "chapter_id": str(chapter_id)})

        # iterate over the transcription candidates
        for t_i, transcription in enumerate(sample["candidates"]):

            transcription_id = f"{text_id}.s{sentence_cnt:03d}"

            transcription_attributes = {"id": transcription_id,
                                        "transcription": transcription,
                                        "sentence_id": str(sample["id"]),
                                        "speaker_id": str(sample["speaker_id"]),
                                        "lm_probability": str(sample["lm_probability"][t_i]),
                                        "logit_probability": str(sample["logit_probability"][t_i])}

            sentence = ET.SubElement(text, "sentence", attrib=transcription_attributes)
            sentence_cnt += 1

            tokenization = tagging_pipeline.process(transcription)

            assert len(tokenization.sentences) == 1, f"Transcription {transcription_id} is splitted in more than" \
                                                     f" 1 sentence by stanza pipeline, we're losing information!!!"

            instance_cnt = 0

            for token in tokenization.sentences[0].words:

                pos, lemma = token.upos, token.lemma
                attributes = {"pos": pos}

                if lemma is not None:
                    attributes["lemma"] = lemma

                if pos in pos_map and lemma is not None and len(synsets_from_lemmapos(lemma, pos_map[pos])) > 1:
                    attributes["id"] = f"{transcription_id}.t{instance_cnt:03d}"
                    word = ET.SubElement(sentence, "instance", attrib=attributes)
                    instance_cnt += 1
                else:
                    word = ET.SubElement(sentence, "wf", attrib=attributes)

                word.text = token.text

        pbar.update(1)

    tree = ET.ElementTree(corpus)
    ET.indent(tree, space="", level=0)

    dump_file_name = f"{model_name}-{dataset_name}.data.xml"
    tree.write(os.path.join(predictions_path, dump_file_name), encoding="UTF-8", xml_declaration=True)
    
    
def compute_wsd_scores(xml_data_path: str, txt_keys_path: str, wsd_ngram_model_path: str, wsd_pmi_attrs_path: str) -> None:
    """
        Computes PMI and LM scores of the senses from the given WSD corpus (computed through `compute_wsd_dataset`),
        based on the predictions of a WSD model contained in the given keys file,
        and dumps a json grouped by their "sentence_id" attribute.

    Args:
        xml_data_path (`str`):
            Path to an xml WSD corpus computed through the `compute_wsd_dataset` function.
        txt_keys_path (`str`):
            Path to a txt labels keys file.
        wsd_ngram_model_path (`str`):
            Path to a dump of a sense-level n-gram model.
        wsd_pmi_attrs_path (`str`):
            Path to a dump of a sense-level PMI's frequences.
    """
    if not os.path.isfile(xml_data_path):
        raise ValueError(f"{xml_data_path} is not a valid xml data file")    

    if not os.path.isfile(wsd_ngram_model_path):
        raise ValueError(f"{wsd_ngram_model_path} is not a valid arpa n-gram file")
    
    if not os.path.isfile(wsd_pmi_attrs_path):
        raise ValueError(f"{wsd_pmi_attrs_path} is not a valid json pmi's dump file")
    
    lemma_keys_dict = read_wsd_keys(txt_keys_path)

    language_model = kenlm.LanguageModel(wsd_ngram_model_path)
    pmi_model = PointwiseMutualInformation.load_from_dir(wsd_pmi_attrs_path)

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
                    lm_score = language_model.score(" ".join(bn_sense_ids))

                    sample["wsd_lm_scores"].append(lm_score)
                    sample["esc_predictions"].append(lemma_key)
                    sample["bn_esc_predictions"].append(bn_sense_id)
                    sample["senses"].append(token_xml.text)
                    sample["sense_indices"].append(token_i)

        for i in range(len(bn_sense_ids)):
            pmi_score = pmi_model.compute_average_pmi(i, bn_sense_ids)
            sample["wsd_pmi_scores"].append(pmi_score)

    scored_dataset_path = os.path.join(os.path.dirname(xml_data_path), f"{stem_basename_suffix(xml_data_path)}_scored")

    with open(f"{scored_dataset_path}.json", "w") as f:
        json.dump(samples, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.set_defaults(mode="preprocess")    

    # required
    preprocess_parser.add_argument("-m", "--model-name", type=str, required=True)
    # default + not required
    preprocess_parser.add_argument("-p", "--predictions-path", type=str, default=PREDICTIONS_PATH)
    # mutually exclusive
    group = preprocess_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--dataset-name", type=str, default=None)
    group.add_argument("-l", "--librispeech", action="store_true")
    
    scores_parser = subparsers.add_parser("scores")
    scores_parser.set_defaults(mode="scores")
    
    # required
    scores_parser.add_argument("--wsd-dataset-path", type=str, required=True)
    scores_parser.add_argument("--wsd-labels-path", type=str, required=True)
    scores_parser.add_argument("--ngram-model-path", type=str, required=True)
    scores_parser.add_argument("--pmi-attrs-path", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "preprocess":

        if args.librispeech:
            compute_wsd_dataset_librispeech(args.model_name)

        if args.dataset_name is not None:
            dataset = datasets.Dataset.load_from_disk(os.path.join(args.predictions_path, args.dataset_name))
            compute_wsd_dataset(dataset, args.dataset_name, args.model_name, args.predictions_path)
    
    elif args.mode == "scores":
        
        compute_wsd_scores(xml_data_path=args.wsd_dataset_path, txt_keys_path=args.wsd_labels_path,
                           wsd_ngram_model_path=args.ngram_model_path, wsd_pmi_attrs_path=args.pmi_attrs_path)
