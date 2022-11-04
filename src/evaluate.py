import argparse
import json
from typing import Dict

import datasets
import wandb
from jiwer import wer, wil
from pprint import pprint

from constants import *
from models import load_pretrained_model, Wav2Vec2WithWSD, Wav2Vec2WithLM
from utils.metrics import PointwiseMutualInformation


def compute_transcriptions_metrics(model_name: str, filtered_predictions_all: datasets.Dataset) -> Dict[str, float]:
    predictions_other = datasets.Dataset.load_from_disk(f"{DATA_PATH}predictions/{model_name}-test_other")
    predictions_clean = datasets.Dataset.load_from_disk(f"{DATA_PATH}predictions/{model_name}-test_clean")
    predictions_all = datasets.concatenate_datasets([predictions_other, predictions_clean])
    
    corrected_transcriptions_cnt = 0
    correct_transcriptions_cnt, wrong_transcriptions_cnt = 0, 0
    wrong_transcriptions_changed_cnt, correct_transcription_filtered_cnt = 0, 0
    transcription_not_in_candidates_cnt, correct_transcription_in_new_candidates_cnt = 0, 0

    for i in range(len(predictions_all)):
        text = predictions_all[i]["text"]
        transcription = predictions_all[i]["transcription"]
        candidates = predictions_all[i]["candidates"]
        new_transcription = filtered_predictions_all[i]["transcription"]
        new_candidates = filtered_predictions_all[i]["candidates"]
        if transcription != text:
            wrong_transcriptions_cnt += 1
            
            if transcription != new_transcription:
                wrong_transcriptions_changed_cnt += 1
                if text == new_transcription:
                    corrected_transcriptions_cnt += 1
            
            if text not in candidates:
                transcription_not_in_candidates_cnt += 1
                if text in new_candidates:
                    correct_transcription_in_new_candidates_cnt += 1

        else:
            correct_transcriptions_cnt += 1
            if transcription != new_transcription:
                correct_transcription_filtered_cnt += 1
                
    return {
        "corrected_transcriptions": corrected_transcriptions_cnt / wrong_transcriptions_cnt,
        "wrong_transcriptions_changed": wrong_transcriptions_changed_cnt / wrong_transcriptions_cnt,        
        "correct_transcription_filtered": correct_transcription_filtered_cnt / correct_transcriptions_cnt,
        "correct_transcription_in_new_candidates": correct_transcription_in_new_candidates_cnt / transcription_not_in_candidates_cnt,
    }
    
    
def compute_wer(predictions: Dict[str, datasets.Dataset]) -> Dict[str, float]:
    clean_wer = wer(predictions["clean"]["text"], predictions["clean"]["transcription"])
    other_wer = wer(predictions["other"]["text"], predictions["other"]["transcription"])
    all_wer = wer(predictions["all"]["text"], predictions["all"]["transcription"])

    clean_wil = wil(predictions["clean"]["text"], predictions["clean"]["transcription"])
    other_wil = wil(predictions["other"]["text"], predictions["other"]["transcription"])
    all_wil = wil(predictions["all"]["text"], predictions["all"]["transcription"])

    return {
        "wer_other": other_wer,
        "wer_clean": clean_wer,
        "wer_all": all_wer,
        "wil_other": other_wil,
        "wil_clean": clean_wil,
        "wil_all": all_wil
    }


def log_on_wandb(config: Dict, run_name: str, predictions: Dict) -> Dict:
    config = {k: v for k, v in config.items() if v is not None}
    wandb.init(project="wsd-s2t", entity="andreagasparini", config=config, name=run_name)

    model_name = f"{config['model'].split('/')[1]}-4-gram"

    metrics = {**compute_wer(predictions), **compute_transcriptions_metrics(model_name, predictions["all"])}

    wandb.log(metrics)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("-m", "--model", type=str, required=True, choices=["base", "large-self"])
    # default + not required
    parser.add_argument("-s", "--scorer", type=str, choices=["lm", "pmi"], default="lm")
    parser.add_argument("-p", "--pmi-mode", type=str, choices=["average", "1-vs-all"], default="average")
    parser.add_argument("-t", "--threshold", type=float, default=None)
    parser.add_argument("-w", "--wandb", action="store_true", default=False)
    parser.add_argument("-d", "--dump", action="store_true", default=False)
    parser.add_argument("--all-min", action="store_true", default=False)
    parser.add_argument("--filter-most-probable-candidates", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()

    BASE = "facebook/wav2vec2-base-960h" # "patrickvonplaten/wav2vec2-base-960h-4-gram"
    LARGE_SELF = "facebook/wav2vec2-large-960h-lv60-self" # "patrickvonplaten/wav2vec2-large-960h-lv60-self-4-gram"    
    
    if args.model == "base":
        MODEL = BASE
    elif args.model == "large-self":
        MODEL = LARGE_SELF

    MODEL_NAME = MODEL.split("/")[1]
    
    TRAIN_CORPORA = [
        "jsonl_wsd",
        # "SemCor",
        # "OMSTI"
    ]
    
    LANGUAGE_MODEL = {
        "type": "WSD Language Model (KenLM)",
        "ngram_size": 4,
        "train_corpora": TRAIN_CORPORA
    }
    
    PMI = {
        "type": "WSD Pointwise Mutual Information",
        "mode": args.pmi_mode,
        "train_corpora": TRAIN_CORPORA
    }
    
    SCORER = PMI if args.scorer == "pmi" else LANGUAGE_MODEL
    
    INCLUDE_MOST_PROBABLE_CANDIDATES = not args.filter_most_probable_candidates
    THRESHOLD = args.threshold
    ARGMIN = not args.all_min

    BS_FILTERING = {
        "criterion": "threshold" if THRESHOLD is not None else "min" if not ARGMIN else "argmin",
        "threshold": THRESHOLD,
        "keep_same_words": True,
        "include_most_probable_candidates": INCLUDE_MOST_PROBABLE_CANDIDATES
    }
    
    scorer = f'-pmi{"-1vsall" if PMI["mode"] == "1-vs-all" else ""}' if SCORER == PMI else '-lm'
    
    ksw = 'ksw2' if BS_FILTERING['keep_same_words'] else ''
    
    if MODEL == BASE:
        if THRESHOLD is not None:
            RUN_NAME=f"wav2vec2-base{scorer}-bs-thresholded-{THRESHOLD}-{ksw}"
        else:
            RUN_NAME=f"wav2vec2-base{scorer}-bs-{'min' if not ARGMIN else 'argmin'}-{ksw}"
    else:
        if THRESHOLD is not None:
            RUN_NAME=f"wav2vec2-large-self{scorer}-bs-thresholded-{THRESHOLD}-{ksw}"
        else:
            RUN_NAME=f"wav2vec2-large-self{scorer}-bs-{'min' if not ARGMIN else 'argmin'}-{ksw}"

    print("=== Loading LibriSpeech test set ===")

    ls_test_other = datasets.Dataset.load_from_disk(f"{DATA_PATH}librispeech/librispeech_test_other")
    ls_test_clean = datasets.Dataset.load_from_disk(f"{DATA_PATH}librispeech/librispeech_test_clean")

    print("Done!")

    print("=== Loading scored LibriSpeech test set ===")

    if "jsonl_wsd" in LANGUAGE_MODEL["train_corpora"]:
        if "SemCor" in LANGUAGE_MODEL["train_corpora"] and "OMSTI" in LANGUAGE_MODEL["train_corpora"]:
            with open(f"{DATA_PATH}predictions/{MODEL_NAME}-4-gram-librispeech_test_all_scored+semcor+omsti.json") as f:
                samples = json.load(f)
        else:
            with open(f"{DATA_PATH}predictions/{MODEL_NAME}-4-gram-librispeech_test_all_scored.json") as f:
                samples = json.load(f)

    print("Done!")

    print("=== Loading pre-trained Wav2Vec 2.0 model ===")
    
    pmi = PointwiseMutualInformation.load_from_dir(f"{MODELS_PATH}pmi/pmi.json") if args.scorer == "pmi" else None
    pretrained_model, pretrained_processor = load_pretrained_model(MODEL)
    model = Wav2Vec2WithWSD(pretrained_model, pretrained_processor, f"{NGRAMS_PATH}librispeech/4-gram", samples,
                            include_most_probable_candidates=INCLUDE_MOST_PROBABLE_CANDIDATES,
                            threshold=THRESHOLD, argmin=ARGMIN, pmi=pmi, pmi_mode=PMI["mode"])
    
    CONFIG = {
        "model": MODEL,
        "scorer": SCORER,
        "decoding": "Beam search" if isinstance(model, Wav2Vec2WithLM) else "Greedy search",
        "beam_search_filtering_settings": BS_FILTERING if SCORER is not None else None
    }

    pprint(CONFIG)

    print("Done!")

    print("=== Computing predictions and filtering beam search ===")
    
    map_function = model.filtered_beam_search

    if THRESHOLD is not None:
        base_save_path = f"{DATA_PATH}predictions/{MODEL_NAME}-4-gram{scorer}-filtered_bs_thresholded_{THRESHOLD}_{ksw}"
    else:
        base_save_path = f"{DATA_PATH}predictions/{MODEL_NAME}-4-gram{scorer}-filtered_bs_{'min' if not ARGMIN else 'argmin'}_{ksw}"

    preds = dict()
    preds["other"] = ls_test_other.map(map_function, remove_columns=["file", "audio"])
    preds["clean"] = ls_test_clean.map(map_function, remove_columns=["file", "audio"])
    preds["all"] = datasets.concatenate_datasets([preds["other"], preds["clean"]])
    
    if args.dump:
        preds["other"].save_to_disk(f"{base_save_path}-test_other")
        preds["clean"].save_to_disk(f"{base_save_path}-test_clean")

    if args.wandb:

        print("=== Storing results on Weights & Biases ===")

        log_on_wandb(config=CONFIG, predictions=preds, run_name=RUN_NAME)

    print("Done!")
    exit(0)
