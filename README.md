
## Usage

### Generate beam search predictions of a Wav2Vec 2.0 + LM model
```python
from tagging import generate_librispeech_predictions
generate_librispeech_predictions("facebook/wav2vec2-base-960h")
```

### Tag the predictions of a Wav2Vec 2.0 + LM with lemma and POS

```bash
python tagging.py preprocess --model-name wav2vec2-base-960h-4-gram --librispeech
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

Where the dataset-paths you provide to the model can be either:
- in a format that follows the one introduced by [Raganato et al. (2017)](https://www.aclweb.org/anthology/E17-1010/)
- jsonl files (or a directory containing them) with one json per line, which must have a `labels` key containing a list of BabelNet identifiers

### Assign LM's scores to the senses of the disambiguated LibriSpeech predictions

```bash
python tagging.py scores \
  --wsd-dataset-path ../data/predictions/wav2vec2-base-960h-4-gram-librispeech_test_all.data.xml \
  --wsd-labels-path ../esc/predictions/wav2vec2-base-960h-4-gram-librispeech_test_all_predictions.txt \
  --ngram-model-path ../models/ngrams/4gram.arpa \
  --pmi-attrs-path ../models/pmi/pmi.json
```
