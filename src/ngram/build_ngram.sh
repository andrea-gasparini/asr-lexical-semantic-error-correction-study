#!bin/bash

NGRAM_PATH="../../models/ngrams/"
SENSE_IDS_FILENAME="sense_ids.txt"

read -rep "Enter JSONL WSD dataset path: " jsonl_wsd_dataset_path
read -rp "Enter the size of the n-gram: " ngram_size

echo "=== Extracting WSD labels from the dataset ==="

cat $jsonl_wsd_dataset_path | python preprocess_wsd_dataset.py # | kenlm/build/bin/lmplz -o $ngram_size > "${NGRAM_PATH}${ngram_size}gram.arpa"

echo "Done!"

kenlm/build/bin/lmplz -o $ngram_size < $NGRAM_PATH$SENSE_IDS_FILENAME > "${NGRAM_PATH}${ngram_size}gram.arpa"

echo "=== Fixing KenLM format as expected in Transformers 🤗 ==="

python postprocess_ngram.py --ngram-size=$ngram_size --data-path=$NGRAM_PATH
