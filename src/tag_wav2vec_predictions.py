import json
from textwrap import indent
from xml.etree import ElementTree as ET
from tqdm import tqdm

import nltk
from nltk.corpus import wordnet
import stanza
import datasets
from datasets import load_dataset

from utils import synsets_from_lemmapos, pos_map

ROOT_DIR = "/media/andrea/512Gb/tesi/"

def main() -> None:

    ls_test_other = datasets.Dataset.load_from_disk(f"{ROOT_DIR}predictions/wav2vec2-large-960h-lv60-self-4-gram"
                                                    f"-test_other_predictions")
    ls_test_clean = datasets.Dataset.load_from_disk(f"{ROOT_DIR}predictions/wav2vec2-large-960h-lv60-self-4-gram"
                                                    f"-test_clean_predictions")
    ls_test_all = datasets.concatenate_datasets([ls_test_clean, ls_test_other])

    ls_test_clean = ls_test_clean.sort(column="chapter_id")
    ls_test_other = ls_test_other.sort(column="chapter_id")
    ls_test_all = ls_test_all.sort(column="chapter_id")

    tagging_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_no_ssplit=True)

    for dataset_name, dataset in [("librispeech_test_other", ls_test_other),
                                  ("librispeech_test_clean", ls_test_clean),
                                  ("librispeech_test_all", ls_test_all)]:

        corpus = ET.Element("corpus", attrib={"lang": "en", "source": dataset_name})

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

            for t_i, transcription in enumerate(sample["transcription"]):

                transcription_id = f"{text_id}.s{sentence_cnt:03d}"

                transcription_attributes = {"id": transcription_id,
                                            "sentence_id": str(sample["id"]),
                                            "speaker_id": str(sample["speaker_id"]),
                                            "lm_probability": str(sample["lm_probability"][t_i]),
                                            "logit_probability": str(sample["logit_probability"][t_i])}

                sentence = ET.SubElement(text, "sentence", attrib=transcription_attributes)
                sentence_cnt += 1

                tokenization = tagging_pipeline.process(transcription)

                assert len(tokenization.sentences) == 1, f"Transcription {transcription_id} is splitted in more than 1 sentence by stanza pipeline, we're losing information!!!"

                instance_cnt = 0

                for token in tokenization.sentences[0].words:

                    pos, lemma = token.upos, token.lemma
                    attributes = { "pos": pos }

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

        # print(ET.dump(tree))

        tree.write(f"{ROOT_DIR}data/librispeech/{dataset_name}.data.xml", encoding="UTF-8", xml_declaration=True)


if __name__ == "__main__":

    main()

    # ls_test_other = datasets.Dataset.load_from_disk(f"{ROOT_DIR}predictions/wav2vec2-large-960h-lv60-self-4-gram"
    #                                                 f"-test_other_predictions")