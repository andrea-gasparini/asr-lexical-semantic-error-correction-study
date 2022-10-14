
## Usage

### Tag the predictions of a Wav2Vec 2.0 + LM with lemma and POS

```bash
python tag_wav2vec_predictions.py
```

### Disambiguate the predictions with a pre-trained model (e.g. [Escher](https://github.com/SapienzaNLP/esc))
```bash
git clone https://github.com/SapienzaNLP/esc.git 
cd esc
bash ./setup.sh
PYTHONPATH=$(pwd) python esc/predict.py --ckpt <escher_checkpoint.ckpt> --dataset-paths ../data/librispeech/librispeech_test_all.data.xml --prediction-types probabilistic
```

### Train an 4-gram Language Model on the senses of a WSD dataset

```bash
python ngram.py --wsd-dataset-paths ../data/WSD_huge_corpus/ --ngram-size 4 --binary
```

### Assign LM's scores to the senses of the disambiguated LibriSpeech predictions

```bash
python ngram_ranking.py \
  --wsd-dataset-path ../data/librispeech/librispeech_test_all.data.xml \
  --wsd-labels-path ../data/librispeech/librispeech_test_all.silver.key.txt \
  --ngram-model-path ../models/ngrams/4gram.arpa
```
