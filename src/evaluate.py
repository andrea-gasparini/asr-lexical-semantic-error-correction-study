import json
import os
from functools import partial
from typing import Dict
from typing import Optional, Union, List
from typing import Tuple

import datasets
import numpy as np
import torch
from jiwer import wer, wil
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

import wandb
from constants import *
from constants import DATA_DIR
from utils import list_to_dict


def log_on_wandb(config: Dict, run_name: str, predictions: Dict) -> Dict:
    config = {k: v for k, v in config.items() if v is not None}
    wandb.init(project="wsd-s2t", entity="andreagasparini", config=config, name=run_name)

    clean_wer = wer(predictions["clean"]["text"], predictions["clean"]["transcription"])
    other_wer = wer(predictions["other"]["text"], predictions["other"]["transcription"])
    all_wer = wer(predictions["all"]["text"], predictions["all"]["transcription"])

    clean_wil = wil(predictions["clean"]["text"], predictions["clean"]["transcription"])
    other_wil = wil(predictions["other"]["text"], predictions["other"]["transcription"])
    all_wil = wil(predictions["all"]["text"], predictions["all"]["transcription"])

    metrics = {
        "wer_other": other_wer,
        "wer_clean": clean_wer,
        "wer_all": all_wer,
        "wil_other": other_wil,
        "wil_clean": clean_wil,
        "wil_all": all_wil,
    }

    wandb.log(metrics)

    return metrics


def compute_beams_to_filter(sample: Dict, debug: bool = False, threshold: Optional[float] = None):
    indices = sample["sense_indices"]
    bn_predictions = sample["bn_esc_predictions"]

    # TODO: we're not considering sequences with less senses than the 1st one

    beams_to_filer = list()

    for i, bn_id in enumerate(bn_predictions[0]):

        ll, l, indices_f, tokens = list(), list(), list(), list()

        for ii, ids in enumerate(bn_predictions):
            if len(ids) > i and indices[0][i] == indices[ii][i] and sample["tokens"][0][indices[0][i]] != \
                    sample["tokens"][ii][indices[ii][i]]:
                l.append(ids[i])
                ll.append(sample["wsd_lm_scores"][ii][i])
                indices_f.append(indices[ii])
                tokens.append(sample["tokens"][ii])

        if any([x for x in l if x != l[0]]):
            if debug: print(l)
            if debug: print(ll)

            if threshold is None:
                min_idx = np.argmin(ll)
                if debug: print(min_idx)
                beams_to_filer.append(" ".join(tokens[min_idx][:indices_f[min_idx][i] + 1]))
            else:
                max_value = max(ll)
                for el_index, el in enumerate(ll):
                    delta = max_value - el
                    if delta > threshold:
                        if debug: print(el_index)
                        beams_to_filer.append(" ".join(tokens[el_index][:indices_f[el_index][i] + 1]))

    return beams_to_filer


def load_pretrained_model(local_dumps_dir: str, hf_model_url: str) \
        -> Tuple[Wav2Vec2ForCTC, Union[Wav2Vec2Processor, Wav2Vec2ProcessorWithLM]]:
    model_name = hf_model_url.split("/")[1]

    local_dump_path = f"{local_dumps_dir}{model_name}"

    # load pretrained model
    if not os.path.isdir(local_dump_path):

        model = AutoModelForCTC.from_pretrained(hf_model_url).to("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(hf_model_url)
        model.save_pretrained(local_dump_path)
        processor.save_pretrained(local_dump_path)

    else:

        model = AutoModelForCTC.from_pretrained(local_dump_path).to("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(local_dump_path)

    return model, processor


# def filter_beam_search(threshold: int, wsd_samples: Dict, samples: Dict, hf_sample):
def filter_beam_search(hf_sample):
    inputs = processor_ngram(hf_sample["audio"]["array"], sampling_rate=16_000, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        logits = model_ngram(**inputs).logits[0]

    if hf_sample["id"] in wsd_samples:

        from pyctcdecode_local import BeamSearchDecoderCTC
        decoder = BeamSearchDecoderCTC.load_from_dir(f"{MODELS_DIR}wav2vec2-large-960h-lv60-self-4-gram")
        beams_to_filer = compute_beams_to_filter(list_to_dict(samples[hf_sample["id"]]), threshold=THRESHOLD)

        output_beams = decoder.decode_beams(logits.cpu().numpy(), beams_to_filter=beams_to_filer)

    else:

        output_beams = processor_ngram.decoder.decode_beams(logits.cpu().numpy())

    candidates = [x[0] for x in output_beams]
    transcription = output_beams[0][0] if len(output_beams) > 0 else ""

    hf_sample["candidates"] = candidates
    hf_sample["transcription"] = transcription

    return hf_sample


def get_wsd_differences_samples(ranked_samples: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    wsd_differences_samples = dict()

    for i, (k, v) in enumerate(ranked_samples.items()):

        if len(v) == 1: continue

        p = [(float(vv["lm_probability"]), float(vv["logit_probability"])) for vv in v]
        max_probability_indices = torch.argmax(torch.tensor([[x[i] for x in p] for i in range(len(p[0]))]), dim=-1)
        if torch.all(torch.tensor([x == max_probability_indices[0] for x in max_probability_indices])):
            continue

        predictions = [vv["esc_predictions"] for vv in v]
        if not torch.all(torch.tensor([e == predictions[0] for e in predictions])):
            wsd_differences_samples[k] = v

    return wsd_differences_samples


if __name__ == "__main__":
    print("=== Loading LibriSpeech test set ===")

    ls_test_other = datasets.Dataset.load_from_disk(f"{DATA_DIR}librispeech/librispeech_test_other")
    ls_test_clean = datasets.Dataset.load_from_disk(f"{DATA_DIR}librispeech/librispeech_test_clean")

    print("Done!")

    print("=== Loading ranked LibriSpeech test set ===")

    LANGUAGE_MODEL = {
        "type": "WSD Language Model (KenLM)",
        "ngram_size": 4,
        "train_corpora": [
            "jsonl_wsd",
            # "SemCor",
            # "OMSTI"
        ]
    }

    if "jsonl_wsd" in LANGUAGE_MODEL["train_corpora"]:
        if "SemCor" in LANGUAGE_MODEL["train_corpora"] and "OMSTI" in LANGUAGE_MODEL["train_corpora"]:
            with open(f"{DATA_DIR}librispeech/librispeech_test_all_ranked_jsonl+semcor+omsti.json") as f:
                samples = json.load(f)
        else:
            with open(f"{DATA_DIR}librispeech/librispeech_test_all_ranked_jsonl.json") as f:
                samples = json.load(f)

    wsd_samples = get_wsd_differences_samples(samples)

    print("Done!")

    print("=== Loading pre-trained Wav2Vec 2.0 model ===")

    # MODEL = "patrickvonplaten/wav2vec2-large-960h-lv60-self-4-gram"
    MODEL = "patrickvonplaten/wav2vec2-base-960h-4-gram"
    model_ngram, processor_ngram = load_pretrained_model(MODELS_DIR, MODEL)
    # model, processor = load_pretrained_model(MODELS_DIR, "facebook/wav2vec2-base-960h")
    # wsdmodel = Wav2Vec2WithWSD(model, processor, f"{MODELS_DIR}ngrams/librispeech/4-gram")

    print("Done!")

    print("=== Computing predictions and filtering beam search ===")

    THRESHOLD = 0.75
    map_function = filter_beam_search
    # map_function = partial(filter_beam_search, THRESHOLD)  # , wsd_samples, samples)
    # map_function = wsdmodel.beam_search

    # base_save_path = f"{DATA_DIR}predictions/filtered_beam_search_thresholded_{THRESHOLD}_nosamewords"
    base_save_path = f"{DATA_DIR}predictions/wav2vec2-base-960h-4-gram"

    predictions = dict()
    predictions["other"] = ls_test_other.map(map_function, remove_columns=["file", "audio"])
    predictions["other"].save_to_disk(f"{base_save_path}-test_other")
    predictions["clean"] = ls_test_clean.map(map_function, remove_columns=["file", "audio"])
    predictions["clean"].save_to_disk(f"{base_save_path}-test_clean")
    predictions["all"] = datasets.concatenate_datasets([predictions["other"], predictions["clean"]])

    print("=== Storing results on Weights & Biases ===")

    SCORER = LANGUAGE_MODEL

    BS_FILTERING = {
        "criterion": "threshold" if THRESHOLD is not None else "argmin",
        "threshold": THRESHOLD,
        "keep_same_words": True
    }

    config = {
        "model": MODEL,
        "scorer": SCORER,
        "decoding": "Beam search" if "gram" in MODEL else "Greedy search",
        "beam_search_filtering_settings": BS_FILTERING if SCORER is not None else None
    }

    log_on_wandb(config=config, predictions=predictions,
                 # run_name=f"wav2vec2-large-self-lm-bs-thresholded-{THRESHOLD}-ksw-alltrainwsdlm")
                 run_name=f"wav2vec2-base-lm-bs-thresholded-{THRESHOLD}-ksw")

    print("Done!")
    exit(0)
