import argparse
import json
import os
from functools import partial
from typing import Dict, Literal
from typing import Optional, Union, List
from typing import Tuple

import datasets
import numpy as np
import torch
import wandb
from jiwer import wer, wil
from pprint import pprint
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM

from pyctcdecode_local import BeamSearchDecoderCTC
from constants import *
from constants import DATA_DIR
from utils.metrics import PointwiseMutualInformation as PMI
from utils import list_to_dict


def compute_transcriptions_metrics(model_name: str, filtered_predictions_all: datasets.Dataset) -> Dict[str, float]:
    predictions_other = datasets.Dataset.load_from_disk(f"{DATA_DIR}predictions/{model_name}-test_other")
    predictions_clean = datasets.Dataset.load_from_disk(f"{DATA_DIR}predictions/{model_name}-test_clean")
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

    model_name = config["model"].split("/")[1]

    metrics = {**compute_wer(predictions), **compute_transcriptions_metrics(model_name, predictions["all"])}

    wandb.log(metrics)

    return metrics


def compute_beams_to_filter_pmi_1_vs_all(sample: Dict, pmi: PMI, threshold: float = 0.0):
    if threshold is None:
        threshold = 0.0
    
    senses_lists = sample["bn_esc_predictions"]
    sense_indices = sample["sense_indices"]
    tokens = sample["tokens"]

    beams_to_filter = list()

    for candidate_idx, senses in enumerate(senses_lists):

        senses = [sense for sense in senses if sense in pmi.unigram_frequences]

        for sense_idx, sense in enumerate(senses):
            is_valid, is_in_pmi = False, False
            for sense_idx2, sense2 in enumerate(senses):
                
                if sense_idx == sense_idx2:
                    continue
                
                if f"{sense} {sense2}" in pmi.bigram_frequences:
                    is_in_pmi = True
                    v = pmi.pmi(sense, sense2)				
                    if v > threshold:
                        is_valid = True
                        break
                else:
                    # TODO can we do better than ignoring? (case w/ unseen pair in the train corpus)
                    continue

            if not is_valid:				
                if is_in_pmi:
                    sense_token_index = sense_indices[candidate_idx][sense_idx]
                    beams_to_filter.append(" ".join(tokens[candidate_idx][:sense_token_index + 1]))
        
    return beams_to_filter


def compute_beams_to_filter(sample: Dict, threshold: Optional[float] = None,
                            mode: Literal["lm", "pmi"] = "lm", pmi: Optional[PMI] = None):
    if mode == "pmi" and pmi is None:
        raise ValueError(f"when `mode` == 'pmi' you must give a valid `pmi` parameter as well")
    
    indices = sample["sense_indices"]
    bn_predictions = sample["bn_esc_predictions"]
    scores = sample["wsd_lm_scores" if mode == "lm" else "wsd_pmi_scores"]
    tokens = sample["tokens"]

    # TODO: we're not considering sequences with less senses than the 1st one

    beams_to_filter = list()

    for i, transcription_bn_id in enumerate(bn_predictions[0]):
        
        if mode == "pmi" and transcription_bn_id not in pmi.unigram_frequences:
            continue
        
        tran_bn_id_idx = indices[0][i]
        tran_tokens = tokens[0]

        bn_ids_window = [transcription_bn_id]
        scores_window = [scores[0][i]]
        indices_window = [indices[0]]
        tokens_window = [tokens[0]]

        for ii, candidate_bn_ids in enumerate(bn_predictions):

            # skip transcription candidate (already inserted)
            if ii == 0:
                continue

            if len(candidate_bn_ids) > i:
                
                if mode == "pmi" and candidate_bn_ids[i] not in pmi.unigram_frequences:
                    continue
                
                cand_bn_id_idx = indices[ii][i]
                cand_tokens = tokens[ii]

                tokens_are_different = tran_tokens[tran_bn_id_idx] != cand_tokens[cand_bn_id_idx]
                bn_ids_are_equal = transcription_bn_id == candidate_bn_ids[i]

                if tran_bn_id_idx == cand_bn_id_idx and (tokens_are_different or bn_ids_are_equal):
                    bn_ids_window.append(candidate_bn_ids[i])
                    scores_window.append(scores[ii][i])
                    indices_window.append(indices[ii])
                    tokens_window.append(tokens[ii])

        if any([id for id in bn_ids_window if id != bn_ids_window[0]]):
            if threshold is None:
                if ARGMIN:
                    min_idx = np.argmin(scores_window)
                    beams_to_filter.append(" ".join(tokens_window[min_idx][:indices_window[min_idx][i] + 1]))
                else:
                    min_value = min(scores_window)
                    for el_index, el in enumerate(scores_window):
                        if el == min_value:
                            beams_to_filter.append(" ".join(tokens_window[el_index][:indices_window[el_index][i] + 1]))
            else:
                max_value = max(scores_window)
                for el_index, el in enumerate(scores_window):
                    delta = max_value - el
                    if delta > threshold:
                        beams_to_filter.append(" ".join(tokens_window[el_index][:indices_window[el_index][i] + 1]))

    return beams_to_filter


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


def forward(hf_sample):
    inputs = processor_ngram(hf_sample["audio"]["array"], sampling_rate=16_000, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        logits = model_ngram(**inputs).logits[0]

    return logits


def filter_beam_search(decoder: BeamSearchDecoderCTC, threshold: int, wsd_samples: Dict, samples: Dict, pmi: PMI, hf_sample):
    if hf_sample["id"] in wsd_samples:

        if SCORER == _PMI:
            if _PMI["mode"] == "average":
                beams_to_filter = compute_beams_to_filter(list_to_dict(samples[hf_sample["id"]]), threshold=threshold, mode="pmi", pmi=pmi)
            else:
                beams_to_filter = compute_beams_to_filter_pmi_1_vs_all(list_to_dict(samples[hf_sample["id"]]), threshold=threshold, pmi=pmi)
        elif SCORER == LANGUAGE_MODEL:
            beams_to_filter = compute_beams_to_filter(list_to_dict(samples[hf_sample["id"]]), threshold=threshold, mode="lm")
        else:
            raise ValueError(f"scorer {SCORER} not supported")

        output_beams = decoder.decode_beams(forward(hf_sample).cpu().numpy(), beams_to_filter=beams_to_filter)
        candidates = [x[0] for x in output_beams]
        transcription = output_beams[0][0] if len(output_beams) > 0 else ""

    else:

        ranked_sample = list_to_dict(samples[hf_sample["id"]])
        if "transcription" in ranked_sample:
            candidates = ranked_sample["transcription"]
            transcription = candidates[0]
        else:
            output_beams = processor_ngram.decoder.decode_beams(forward(hf_sample).cpu().numpy())
            candidates = [x[0] for x in output_beams]
            transcription = output_beams[0][0] if len(output_beams) > 0 else ""

    hf_sample["candidates"] = candidates
    hf_sample["transcription"] = transcription

    return hf_sample

def get_wsd_differences_samples(ranked_samples: Dict[str, List[Dict]],
                                include_most_probable_candidates: bool = True) -> Dict[str, List[Dict]]:
    wsd_differences_samples = dict()

    for i, (k, v) in enumerate(ranked_samples.items()):

        # skip samples w/ only one candidate transcription
        if len(v) == 1: continue

        # skip samples w/ the picked transcription being the most probable for both LM and ASR models
        if not include_most_probable_candidates:
            p = [(float(vv["lm_probability"]), float(vv["logit_probability"])) for vv in v]
            max_probability_indices = torch.argmax(torch.tensor([[x[i] for x in p] for i in range(len(p[0]))]), dim=-1)
            if torch.all(torch.tensor([x == max_probability_indices[0] for x in max_probability_indices])):
                continue

        # take into consideration only samples w/ at least one candidate transcription
        # containing different BabelNet identifiers w/ respect to the other candidates
        predictions = [vv["esc_predictions"] for vv in v]
        if not torch.all(torch.tensor([e == predictions[0] for e in predictions])):
            wsd_differences_samples[k] = v

    return wsd_differences_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("-m", "--model", type=str, required=True, choices=["base", "large-self"])
    # default + not required
    parser.add_argument("-s", "--scorer", type=str, choices=["lm", "pmi"], default="lm")
    parser.add_argument("-p", "--pmi-mode", type=str, choices=["average", "1-vs-all"], default="average")
    parser.add_argument("-t", "--threshold", type=float, default=None)
    parser.add_argument("-w", "--wandb", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()

    BASE = "patrickvonplaten/wav2vec2-base-960h-4-gram"
    LARGE_SELF = "patrickvonplaten/wav2vec2-large-960h-lv60-self-4-gram"    
    
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
    
    _PMI = {
        "type": "WSD Pointwise Mutual Information",
        "mode": args.pmi_mode,
        "train_corpora": TRAIN_CORPORA
    }
    
    SCORER = _PMI if args.scorer == "pmi" else LANGUAGE_MODEL
    
    INCLUDE_MOST_PROBABLE_CANDIDATES: bool = True
    THRESHOLD = args.threshold
    ARGMIN = True

    BS_FILTERING = {
        "criterion": "threshold" if THRESHOLD is not None else "min" if not ARGMIN else "argmin",
        "threshold": THRESHOLD,
        "keep_same_words": True,
        "include_most_probable_candidates": INCLUDE_MOST_PROBABLE_CANDIDATES
    }
    
    CONFIG = {
        "model": MODEL,
        "scorer": SCORER,
        "decoding": "Beam search" if "gram" in MODEL else "Greedy search",
        "beam_search_filtering_settings": BS_FILTERING if SCORER is not None else None
    }
    
    pprint(CONFIG)
    
    scorer = f'-pmi-{"1vsall" if _PMI["mode"] == "1-vs-all" else ""}' if SCORER == _PMI else '-lm'
    
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

    ls_test_other = datasets.Dataset.load_from_disk(f"{DATA_DIR}librispeech/librispeech_test_other")
    ls_test_clean = datasets.Dataset.load_from_disk(f"{DATA_DIR}librispeech/librispeech_test_clean")

    print("Done!")

    print("=== Loading ranked LibriSpeech test set ===")

    if "jsonl_wsd" in LANGUAGE_MODEL["train_corpora"]:
        if "SemCor" in LANGUAGE_MODEL["train_corpora"] and "OMSTI" in LANGUAGE_MODEL["train_corpora"]:
            with open(f"{DATA_DIR}predictions/{MODEL_NAME}-librispeech_test_all_ranked+semcor+omsti.json") as f:
                samples = json.load(f)
        else:
            with open(f"{DATA_DIR}predictions/{MODEL_NAME}-librispeech_test_all_ranked.json") as f:
                samples = json.load(f)

    wsd_samples = get_wsd_differences_samples(samples, INCLUDE_MOST_PROBABLE_CANDIDATES)

    print("Done!")

    print("=== Loading pre-trained Wav2Vec 2.0 model ===")

    model_ngram, processor_ngram = load_pretrained_model(MODELS_DIR, MODEL)
    # model, processor = load_pretrained_model(MODELS_DIR, "facebook/wav2vec2-base-960h")
    # wsdmodel = Wav2Vec2WithWSD(model, processor, f"{MODELS_DIR}ngrams/librispeech/4-gram")

    print("Done!")

    print("=== Computing predictions and filtering beam search ===")
    
    decoder = BeamSearchDecoderCTC.load_from_dir(f"{MODELS_DIR}ngrams/librispeech/4-gram")

    pmi = PMI.load_from_dir(f"{MODELS_DIR}pmi/jsonl.json")

    map_function = partial(filter_beam_search, decoder, THRESHOLD, wsd_samples, samples, pmi)
    # map_function = wsdmodel.beam_search

    if THRESHOLD is not None:
        base_save_path = f"{DATA_DIR}predictions/{MODEL_NAME}{scorer}-filtered_bs_thresholded_{THRESHOLD}_{ksw}"
    else:
        base_save_path = f"{DATA_DIR}predictions/{MODEL_NAME}{scorer}-filtered_bs_{'min' if not ARGMIN else 'argmin'}_{ksw}"

    preds = dict()
    preds["other"] = ls_test_other.map(map_function, remove_columns=["file", "audio"])
    preds["other"].save_to_disk(f"{base_save_path}-test_other")
    preds["clean"] = ls_test_clean.map(map_function, remove_columns=["file", "audio"])
    preds["clean"].save_to_disk(f"{base_save_path}-test_clean")
    preds["all"] = datasets.concatenate_datasets([preds["other"], preds["clean"]])

    if args.wandb:

        print("=== Storing results on Weights & Biases ===")

        log_on_wandb(config=CONFIG, predictions=preds, run_name=RUN_NAME)

    print("Done!")
    exit(0)
