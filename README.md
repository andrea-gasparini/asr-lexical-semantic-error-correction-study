
## Usage

### Generate beam search predictions of a Wav2Vec 2.0 + LM model
```bash
```

### Tag the predictions of a Wav2Vec 2.0 + LM with lemma and POS

```bash
python tag_wav2vec_predictions.py --model-name wav2vec2-base-960h-4-gram
```

### Disambiguate the predictions with a pre-trained model (e.g. [Escher](https://github.com/SapienzaNLP/esc))
```bash
git clone https://github.com/SapienzaNLP/esc.git && cd esc && bash setup.sh
```
```bash
PYTHONPATH=$(pwd) python esc/predict.py \
  --ckpt <escher_checkpoint.ckpt> \
  --dataset-paths ../data/predictions/wav2vec2-base-960h-4-gram-librispeech_test_all.data.xml \
  --prediction-types probabilistic
```

### Train an 4-gram Language Model on the senses of a WSD dataset

```bash
python ngram.py --wsd-dataset-paths ../data/WSD_huge_corpus/ --ngram-size 4 --binary
```

### Assign LM's scores to the senses of the disambiguated LibriSpeech predictions

```bash
python ngram_ranking.py \
  --wsd-dataset-path ../data/predictions/wav2vec2-base-960h-4-gram-librispeech_test_all.data.xml \
  --wsd-labels-path ../esc/predictions/wav2vec2-base-960h-4-gram-librispeech_test_all_predictions.txt \
  --ngram-model-path ../models/ngrams/4gram.arpa
```
