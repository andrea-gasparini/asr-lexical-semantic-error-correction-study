import argparse
import json
import os
from typing import List, Literal
from xml.etree import ElementTree as ET

from tqdm import tqdm

from constants import *
from utils.wsd import read_wsd_keys, SenseInventory
from utils.os import ext_is_in_dir, get_num_lines, get_basename


def extract_bn_ids_raganato(sense_ids_file_path: str, xml_data_path: str, txt_keys_path: str,
                            write_mode: Literal["a", "w"] = "w") -> None:
    """
    Extracts the lemma keys' sequences of each sample from a WSD dataset (following Raganato's format),
    maps them to the corresponding BabelNet ids and writes them to a file, one sequence per line.
    Skips sentences which do not contain any tagged sense.
    
    The required format for the WSD dataset has been introduced by Raganato et al. (2017) in
    [Word Sense Disambiguation: A Unified Evaluation Framework and Empirical Comparison](https://www.aclweb.org/anthology/E17-1010/).

    Args:
        sense_ids_file_path (`str`):
            Path to the directory in which to save the file with the extracted BabelNet ids.
        xml_data_path (`str`):
            Path to an xml WSD corpus.
        txt_keys_path (`str`):
            Path to a txt labels keys file.
        write_mode (`Literal["a", "w"]`, optional, defaults to "w"):
            Opening file mode, either "a" for appending or "w" for writing from scratch
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


def extract_bn_ids_jsonl(sense_ids_file_path: str, jsonl_wsd_dataset_path: str,
                         write_mode: Literal["a", "w"] = "w", group_by_document: bool = False) -> None:
    """
    Extracts the sense ids' sequences of each sample from a WSD jsonl dataset and writes them to a file,
    one sequence per line.
    Additional information (e.g. lemma and pos) encoded in the sense ids are filtered in order to only keep
    the actual BabelNet ids, e.g. "bn:00066514n#receptor#NOUN" becomes "bn:00066514n".

    Args:
        sense_ids_file_path (`str`):
            Path to the directory in which to save the file with the extracted BabelNet ids.
        jsonl_wsd_dataset_path (`str`):
            Either a path to a directory containing batches of the dataset or to a single jsonl file.
        write_mode (`Literal["a", "w"]`, optional, defaults to "w"):
            Opening file mode, either "a" for appending or "w" for writing from scratch.
        group_by_document (`bool`):
            Whether to group the annotations by documents instead of sentences.
    """
    with open(sense_ids_file_path, write_mode) as file:

        if os.path.isdir(jsonl_wsd_dataset_path):
            batches = [os.path.join(jsonl_wsd_dataset_path, batch) for batch in os.listdir(jsonl_wsd_dataset_path)]
        elif os.path.isfile(jsonl_wsd_dataset_path):
            batches = [jsonl_wsd_dataset_path]
        else:
            raise ValueError(f"{jsonl_wsd_dataset_path} is neither a directory nor a file")
        
        if group_by_document:
            document_sense_ids: List[str] = list()
            document_id = None

        for batch in batches:

            batch_abspath = os.path.abspath(batch)
            with open(batch_abspath, "r") as wsd_data:
                
                p_bar = tqdm(wsd_data, total=get_num_lines(batch_abspath))
                p_bar.set_description(f"Processing {get_basename(batch)}")

                for jsonl_sample in p_bar:
                    jsonl_sample = json.loads(jsonl_sample)

                    labels: List[str] = list()
                    if "labels" in jsonl_sample:
                        labels = [sense_id[0] for sense_id in jsonl_sample["labels"]]
                    elif "annotations" in jsonl_sample:
                        labels = [sense_id[2] for sense_id in jsonl_sample["annotations"]]
                    else:
                        raise ValueError("sample does not have neither a valid 'labels' nor 'annotations' key")
                    
                    # remove lemma and POS from the id (only keeping "bn:{offset}{wn_pos}")
                    sense_ids = [label.split("#")[0] for label in labels]
                    
                    if group_by_document:

                        if document_id is None:
                            document_id = jsonl_sample["document_id"]
                        
                        if document_id == jsonl_sample["document_id"]:
                            document_sense_ids += sense_ids
                            continue
                        else:
                            sense_ids = [document_id] + document_sense_ids
                            document_id = jsonl_sample["document_id"]
                            document_sense_ids = list()

                    joined_sense_ids = " ".join(sense_ids)
                    file.write(f"{joined_sense_ids}\n")


def extract_bn_ids(wsd_dataset_paths: List[str], sense_ids_file_path: str, group: bool = False) -> None:
    """
    Extracts the BabelNet ids' sequences of each sample from the given WSD datasets and writes them to a file, one per line.

    Args:
        wsd_dataset_paths (`List[str]`):
            A list of paths to either datasets following Raganato's format, or to jsonl ones or to a directory containing
            jsonl batches of the dataset.
        sense_ids_file_path (`str`):
            Path to the directory in which to save the file with the extracted BabelNet ids.
        group (`bool`):
            Whether to group the annotations by documents instead of sentences.
    """
    for i in range(len(wsd_dataset_paths)):

        dataset_path = wsd_dataset_paths[i]
        write_mode = "a" if i > 0 else "w"

        if os.path.isfile(dataset_path) or ext_is_in_dir(".jsonl", dataset_path, "all"):
            extract_bn_ids_jsonl(sense_ids_file_path, dataset_path, write_mode, group)

        elif (len(os.listdir(dataset_path)) == 2 and ext_is_in_dir(".data.xml", dataset_path)
                and ext_is_in_dir(".key.txt", dataset_path)):
            entries = [os.path.join(dataset_path, entry) for entry in os.listdir(dataset_path)]
            data_path, keys_path = entries if entries[0].endswith(".data.xml") else reversed(entries)
            extract_bn_ids_raganato(sense_ids_file_path, data_path, keys_path, write_mode)

        else:
            raise ValueError(f"{dataset_path} is not a valid directory to a Raganato or jsonl WSD dataset")
        

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("-d", "--wsd-dataset-paths", type=str, nargs="+", required=True)
    parser.add_argument("-o", "--sense-ids-path", type=str, required=True)
    # optional + not required
    parser.add_argument("-g", "--group", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_bn_ids(args.wsd_dataset_paths, args.sense_ids_path, args.group)