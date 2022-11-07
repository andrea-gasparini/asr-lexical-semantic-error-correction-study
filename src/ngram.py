import argparse
import os

from tqdm import tqdm

from constants import *
from extract_bn_ids import extract_bn_ids
from utils import get_basename, get_num_lines, stem_basename_suffix


def check_kenlm_setup(kenlm_bin_path: str = KENLM_BIN_PATH) -> None:
    """
    Verifies that a build directory for KenLM already exists, runs the setup script otherwise.

    Args:
        kenlm_bin_path (`str`, optional, defaults to `KENLM_BIN_PATH`):
            Path to the build directory of KenLM.
    """
    if not os.path.isdir(kenlm_bin_path):
        os.system(f"bash {KENLM_SETUP_SCRIPT_PATH}")


def build_ngram(ngram_size: int, ngram_file_path: str, source_txt_file_path: str,
                kenlm_bin_path: str = KENLM_BIN_PATH) -> None:
    """
    Runs the `lmplz` KenLM's script to build a n-gram Language Model and dump it to a .arpa file.

    Args:
        ngram_size (`int`):
            Size of the n-gram to build.
        ngram_file_path (`str`):
            Path to the LM's dump file name we want to create, comprising the .arpa extension.
        source_txt_file_path (`str`):
            Path to the source txt file, containing the tokens of one sample per line.
        kenlm_bin_path (`str`, optional, defaults to `KENLM_BIN_PATH`):
            Path to the build directory of KenLM.
    """
    check_kenlm_setup(kenlm_bin_path)
    os.system(f"{kenlm_bin_path}lmplz -T /tmp -o {ngram_size} < {source_txt_file_path} > {ngram_file_path}")


def build_binary_ngram(ngram_file_path: str, kenlm_bin_path: str = KENLM_BIN_PATH) -> None:
    """
    Runs the `build_binary` KenLM's script to compress a .arpa n-gram Language Model into a binary file. 

    Args:
        ngram_file_path (`str`):
            Path to the LM's .arpa file from which to build the binary one.
        kenlm_bin_path (`str`, optional, defaults to `KENLM_BIN_PATH`):
            Path to the build directory of KenLM.
    """
    check_kenlm_setup(kenlm_bin_path)
    binary_ngram_file_name = f"{stem_basename_suffix(ngram_file_path)}.binary"
    binary_ngram_file_path = os.path.join(os.path.dirname(ngram_file_path), binary_ngram_file_name)
    os.system(f"{kenlm_bin_path}build_binary -T /tmp {ngram_file_path} {binary_ngram_file_path}")
    

def postprocess_ngram(ngram_file_path: str) -> str:
    """
    Post-processes the .arpa n-gram from the given directory in order to use the n-gram in 🤗 Transformers.
    The KenLM n-gram correctly includes an "Unknown" or <unk>, as well as a begin-of-sentence, <s> token,
    but no end-of-sentence, </s> token; which is needed in the Transformers library.

    We simply add the end-of-sentence token by adding the line "0 </s> $begin_of_sentence_score" below the <s> token
    and increasing the n-gram 1 count by 1.
    
    Args:
        ngram_file_path (`str`):
            Path to the .arpa n-gram file.

    Returns:
        `str`:
            Path to the newly created .arpa n-gram file.
    """

    postprocessed_file_path = f"{stem_basename_suffix(ngram_file_path)}_hf.arpa"

    with open(ngram_file_path, "r") as read_file, open(postprocessed_file_path, "w") as write_file:

        has_added_eos = False

        for line in tqdm(read_file, total=get_num_lines(ngram_file_path)):

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("KenLM sense-level n-gram generator")
    # required
    parser.add_argument("--wsd-dataset-paths", type=str, nargs="+", required=True)
    # default + not required
    parser.add_argument("--ngram-path", type=str, default=NGRAMS_PATH)
    parser.add_argument("-s", "--ngram-size", type=int, default=NGRAM_SIZE)
    parser.add_argument("-b", "--binary", action="store_true", default=False)
    parser.add_argument("-p", "--post-process", action="store_true", default=False)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ngram_size = args.ngram_size
    ngram_path = args.ngram_path
    ngram_file_path = os.path.join(ngram_path, f"{ngram_size}gram.arpa")
    sense_ids_file_path = os.path.join(ngram_path, SENSE_IDS_FILENAME)

    check_kenlm_setup()

    print("=== Extracting WSD labels from the datasets ===")

    extract_bn_ids(args.wsd_dataset_paths, sense_ids_file_path)

    build_ngram(ngram_size, ngram_file_path, sense_ids_file_path)

    os.remove(sense_ids_file_path)

    if args.binary:
        print(f"=== Building binary n-gram from {get_basename(ngram_file_path)} ===")
        build_binary_ngram(ngram_file_path)    

    if args.post_process:
        print("=== Fixing KenLM format as expected in 🤗 Transformers ===")
        hf_ngram_file_path = postprocess_ngram(ngram_file_path)

        if args.binary:
            print(f"=== Building binary n-gram from {get_basename(hf_ngram_file_path)} ===")
            build_binary_ngram(hf_ngram_file_path)


if __name__ == "__main__":
    main()
