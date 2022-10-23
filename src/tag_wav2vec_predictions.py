import argparse
import os
from xml.etree import ElementTree as ET

import datasets
import stanza
from tqdm import tqdm

from constants import *
from utils import synsets_from_lemmapos, pos_map


def tag_predictions(dataset: datasets.Dataset, dataset_name: str, predictions_dir: str, model_name: str) -> None:
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

    dump_file_name = f"{predictions_dir}{model_name}-{dataset_name}.data.xml"
    tree.write(dump_file_name, encoding="UTF-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("-m", "--model-name", type=str, required=True)
    # default + not required
    parser.add_argument("-p", "--predictions-path", type=str, default=f"{DATA_DIR}predictions/")
    # mutually exclusive
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--dataset-name", type=str, default=None)
    group.add_argument("-l", "--librispeech", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    predictions_path = args.predictions_path
    model_name = args.model_name

    if args.librispeech:
        ls_test_other = datasets.Dataset.load_from_disk(f"{predictions_path}{model_name}-test_other")
        ls_test_clean = datasets.Dataset.load_from_disk(f"{predictions_path}{model_name}-test_clean")
        ls_test_all = datasets.concatenate_datasets([ls_test_clean, ls_test_other]).sort(column="chapter_id")
        tag_predictions(ls_test_all, "librispeech_test_all", predictions_path, model_name)

    if args.dataset_name is not None:
        dataset = datasets.Dataset.load_from_disk(os.path.join(predictions_path, args.dataset_name))
        tag_predictions(dataset, args.dataset_name, predictions_path, model_name)
