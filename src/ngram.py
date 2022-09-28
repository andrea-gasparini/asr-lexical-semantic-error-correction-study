import json
import os
import argparse
from tqdm import tqdm
from constants import *

from utils import get_num_lines


def extract_wsd_labels(sense_ids_file_path: str, jsonl_wsd_dataset_path: str) -> None:
    """
    Extracts the sense ids' sequences of each sample from a WSD jsonl dataset and writes them to a file, one per line.

    Args:
        sense_ids_file_path: a path where to save the file with the extracted sense ids  
        jsonl_wsd_dataset_path: either a path to a directory containing batches of the dataset or to a single data file
    """

    with open(sense_ids_file_path, "w") as file:

        if os.path.isdir(jsonl_wsd_dataset_path):
            batches = [os.path.join(jsonl_wsd_dataset_path, batch) for batch in os.listdir(jsonl_wsd_dataset_path)]
        elif os.path.isfile(jsonl_wsd_dataset_path):
            batches = [jsonl_wsd_dataset_path]
        else:
            raise ValueError(f"{jsonl_wsd_dataset_path} is neither a directory nor a file")

        for batch in tqdm(batches, total=len(batches)):

            with open(os.path.abspath(batch), "r") as wsd_data:

                for jsonl_sample in wsd_data:

                    jsonl_sample = json.loads(jsonl_sample)

                    assert "labels" in jsonl_sample, f"Sample does not have a valid 'labels' key"

                    sense_ids = jsonl_sample["labels"]
                    joined_sense_ids = " ".join([sense_id[0] for sense_id in sense_ids])

                    file.write(f"{joined_sense_ids}\n")


def postprocess_ngram(ngram_path: str, ngram_size: int) -> None:
    """
    Post-processes the .arpa n-gram file of size `ngram_size` from the directory `ngram_path`.

    This is necessary in order to use the n-gram with 🤗 Transformers.
    The KenLM n-gram correctly includes a "Unknown" or <unk>, as well as a begin-of-sentence, <s> token,
    but no end-of-sentence, </s> token.

    We simply add the end-of-sentence token by adding the line "0 </s> $begin_of_sentence_score" below the <s> token
    and increasing the n-gram 1 count by 1.
    """

    file_prefix = f"{ngram_path}{ngram_size}gram"

    with open(f"{file_prefix}.arpa", "r") as read_file, open(f"{file_prefix}_hf.arpa", "w") as write_file:

        has_added_eos = False

        for line in tqdm(read_file, total=get_num_lines(f"{file_prefix}.arpa")):

            if not has_added_eos and "ngram 1=" in line:
                count=line.strip().split("=")[-1]
                write_file.write(line.replace(f"{count}", f"{int(count)+1}"))

            elif not has_added_eos and "<s>" in line:
                write_file.write(line)
                write_file.write(line.replace("<s>", "</s>"))
                has_added_eos = True

            else:
                write_file.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("KenLM WSD n-gram generator for 🤗 Transformers")
    # required
    parser.add_argument("--wsd-dataset-path", type=str, required=True)
    # default + not required
    parser.add_argument("--ngram-path", type=str, default=NGRAM_PATH)
    parser.add_argument("--ngram-size", type=int, default=NGRAM_SIZE)

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    ngram_size = args.ngram_size
    ngram_file_path = f"{args.ngram_path}{ngram_size}gram.arpa"
    sense_ids_file_path = f"{args.ngram_path}{SENSE_IDS_FILENAME}"

    if not os.path.isdir(KENLM_BIN_PATH):
        os.system("bash setup_kenlm.sh")

    print("=== Extracting WSD labels from the dataset ===")

    extract_wsd_labels(sense_ids_file_path=sense_ids_file_path, jsonl_wsd_dataset_path=args.wsd_dataset_path)

    os.system(f"{KENLM_BIN_PATH}lmplz -o {ngram_size} < {sense_ids_file_path} > {ngram_file_path}")

    os.remove(sense_ids_file_path)

    print("=== Fixing KenLM format as expected in 🤗 Transformers ===")

    postprocess_ngram(ngram_path=args.ngram_path, ngram_size=ngram_size)


if __name__ == "__main__":

    main()
