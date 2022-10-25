import argparse
import json
import os
from typing import List, Literal
from xml.etree import ElementTree as ET

from tqdm import tqdm

from constants import *
from utils import get_basename, get_num_lines, stem_basename_suffix
from utils.os import ext_is_in_dir
from utils.wsd import read_wsd_keys, SenseInventory


def extract_wsd_labels_raganato(sense_ids_file_path: str, xml_data_path: str, txt_keys_path: str,
                                write_mode: Literal["a", "w"] = "w") -> None:
    """
    Extracts the sense ids' sequences of each sample from a WSD dataset (following Raganato's format)
    and writes them to a file, one per line. Skipping sentences which do not contain any tagged sense.
    
    The required format for the WSD dataset has been introduced by Raganato et al. (2017) in
    [Word Sense Disambiguation: A Unified Evaluation Framework and Empirical Comparison](https://www.aclweb.org/anthology/E17-1010/).

    Args:
        sense_ids_file_path: a path where to save the file with the extracted sense ids
        xml_data_path: xml data file
        txt_keys_path: txt keys file
        write_mode: opening file mode, either "a" for appending or "w" for writing from scratch
    """
    if not os.path.isfile(xml_data_path):
        raise ValueError(f"{xml_data_path} is not a valid xml data file")

    lemma_keys_dict = read_wsd_keys(txt_keys_path)

    inventory = SenseInventory()

    with open(sense_ids_file_path, write_mode) as file:
        # iterate over <sentence> tags from the given xml file
        for sent_xml in ET.parse(xml_data_path).iter('sentence'):
            sense_ids: List[str] = list()
            # for each inner xml token (either <instance> or <wf>)
            for token_xml in sent_xml:
                # consider only instance tokens (ignore all <wf>)
                if token_xml.tag == "instance":
                    token_id = token_xml.attrib.get("id")
                    lemma_key = lemma_keys_dict.get(token_id, None)
                    bn_sense_id = inventory.lemma_key_to_bn_id(lemma_key)
                    sense_ids.append(bn_sense_id)

            if len(sense_ids) > 0:
                joined_sense_ids = " ".join(sense_ids)
                file.write(f"{joined_sense_ids}\n")


def extract_wsd_labels_jsonl(sense_ids_file_path: str, jsonl_wsd_dataset_path: str,
                             write_mode: Literal["a", "w"] = "w") -> None:
    """
    Extracts the sense ids' sequences of each sample from a WSD jsonl dataset and writes them to a file, one per line.

    Args:
        sense_ids_file_path: a path where to save the file with the extracted sense ids
        jsonl_wsd_dataset_path: either a path to a directory containing batches of the dataset or to a single jsonl file
        write_mode: opening file mode, either "a" for appending or "w" for writing from scratch
    """
    with open(sense_ids_file_path, write_mode) as file:

        if os.path.isdir(jsonl_wsd_dataset_path):
            batches = [os.path.join(jsonl_wsd_dataset_path, batch) for batch in os.listdir(jsonl_wsd_dataset_path)]
        elif os.path.isfile(jsonl_wsd_dataset_path):
            batches = [jsonl_wsd_dataset_path]
        else:
            raise ValueError(f"{jsonl_wsd_dataset_path} is neither a directory nor a file")

        for batch in tqdm(batches):

            with open(os.path.abspath(batch), "r") as wsd_data:

                for jsonl_sample in wsd_data:
                    jsonl_sample = json.loads(jsonl_sample)

                    assert "labels" in jsonl_sample, f"Sample does not have a valid 'labels' key"

                    labels: List[str] = [sense_id[0] for sense_id in jsonl_sample["labels"]]
                    # remove lemma and POS from the id (only keeping "bn:{offset}{wn_pos}")
                    sense_ids = [label.split("#")[0] for label in labels]
                    joined_sense_ids = " ".join(sense_ids)

                    file.write(f"{joined_sense_ids}\n")


def postprocess_ngram(ngram_path: str, ngram_size: int) -> str:
    """
    Post-processes the .arpa n-gram file of size `ngram_size` from the directory `ngram_path`.

    This is necessary in order to use the n-gram with 🤗 Transformers.
    The KenLM n-gram correctly includes an "Unknown" or <unk>, as well as a begin-of-sentence, <s> token,
    but no end-of-sentence, </s> token.

    We simply add the end-of-sentence token by adding the line "0 </s> $begin_of_sentence_score" below the <s> token
    and increasing the n-gram 1 count by 1.

    Returns:
        the path to the newly created .arpa n-gram file
    """

    file_prefix = f"{ngram_path}{ngram_size}gram"
    postprocessed_file_path = f"{file_prefix}_hf.arpa"

    with open(f"{file_prefix}.arpa", "r") as read_file, open(postprocessed_file_path, "w") as write_file:

        has_added_eos = False

        for line in tqdm(read_file, total=get_num_lines(f"{file_prefix}.arpa")):

            if not has_added_eos and "ngram 1=" in line:
                count = line.strip().split("=")[-1]
                write_file.write(line.replace(f"{count}", f"{int(count) + 1}"))

            elif not has_added_eos and "<s>" in line:
                write_file.write(line)
                write_file.write(line.replace("<s>", "</s>"))
                has_added_eos = True

            else:
                write_file.write(line)

    return postprocessed_file_path


def check_kenlm_setup(kenlm_bin_path: str = KENLM_BIN_PATH) -> None:
    if not os.path.isdir(kenlm_bin_path):
        os.system(f"bash {KENLM_SETUP_SCRIPT_PATH}")


def build_ngram(ngram_size: int, ngram_file_path: str, sense_ids_file_path: str,
                kenlm_bin_path: str = KENLM_BIN_PATH) -> None:
    check_kenlm_setup(kenlm_bin_path)
    os.system(f"{kenlm_bin_path}lmplz -o {ngram_size} < {sense_ids_file_path} > {ngram_file_path}")


def build_binary_ngram(ngram_file_path: str, kenlm_bin_path: str = KENLM_BIN_PATH) -> None:
    check_kenlm_setup(kenlm_bin_path)
    ngram_path = os.path.dirname(ngram_file_path)
    binary_ngram_file_name = f"{stem_basename_suffix(ngram_file_path)}.binary"
    os.system(f"{kenlm_bin_path}build_binary -T /tmp {ngram_file_path} {ngram_path}/{binary_ngram_file_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("KenLM WSD n-gram generator for 🤗 Transformers")
    # required
    parser.add_argument("--wsd-dataset-paths", type=str, nargs="+", required=True)
    # default + not required
    parser.add_argument("--ngram-path", type=str, default=NGRAM_PATH)
    parser.add_argument("-s", "--ngram-size", type=int, default=NGRAM_SIZE)
    parser.add_argument("-b", "--binary", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ngram_size = args.ngram_size
    ngram_path = args.ngram_path
    ngram_file_path = f"{ngram_path}{ngram_size}gram.arpa"
    sense_ids_file_path = f"{ngram_path}{SENSE_IDS_FILENAME}"

    check_kenlm_setup()

    print("=== Extracting WSD labels from the datasets ===")

    for i in tqdm(range(len(args.wsd_dataset_paths))):
        dataset_path = args.wsd_dataset_paths[i]
        write_mode = "a" if i > 0 else "w"
        if os.path.isfile(dataset_path) or ext_is_in_dir(".jsonl", dataset_path, "all"):
            extract_wsd_labels_jsonl(sense_ids_file_path=sense_ids_file_path, jsonl_wsd_dataset_path=dataset_path,
                                     write_mode=write_mode)
        elif (len(os.listdir(dataset_path)) == 2 and ext_is_in_dir(".data.xml", dataset_path)
                and ext_is_in_dir(".key.txt", dataset_path)):
            entries = [os.path.join(dataset_path, entry) for entry in os.listdir(dataset_path)]
            data_path, keys_path = entries if entries[0].endswith(".data.xml") else reversed(entries)
            extract_wsd_labels_raganato(sense_ids_file_path=sense_ids_file_path, xml_data_path=data_path,
                                        txt_keys_path=keys_path, write_mode=write_mode)
        else:
            raise ValueError(f"{dataset_path} is not a valid directory to a Raganato or jsonl WSD dataset")

    build_ngram(ngram_size, ngram_file_path, sense_ids_file_path)

    os.remove(sense_ids_file_path)

    if args.binary:
        print(f"=== Building binary n-gram from {get_basename(ngram_file_path)} ===")
        build_binary_ngram(ngram_file_path)

    print("=== Fixing KenLM format as expected in 🤗 Transformers ===")

    hf_ngram_file_path = postprocess_ngram(ngram_path=ngram_path, ngram_size=ngram_size)

    if args.binary:
        print(f"=== Building binary n-gram from {get_basename(hf_ngram_file_path)} ===")
        build_binary_ngram(hf_ngram_file_path)


if __name__ == "__main__":
    main()
