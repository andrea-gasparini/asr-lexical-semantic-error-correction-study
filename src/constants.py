DATA_DIR: str = "../data/"
MODELS_DIR: str = "../models/"

NGRAM_SIZE: int = 4
NGRAM_PATH: str = f"{MODELS_DIR}ngrams/"
KENLM_BIN_PATH: str = "kenlm/build/bin/"
KENLM_SETUP_SCRIPT_PATH: str = "setup_kenlm.sh"
SENSE_IDS_FILENAME: str = "sense_ids.txt"

LEMMA2IDS_FILE_PATH: str = f"{DATA_DIR}sense_key2wn_id.tsv"
WN2BN_IDS_FILE_PATH: str = f"{DATA_DIR}wn_bn5.tsv"
