import argparse
from collections import Counter
import datasets, torch

from constants import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, choices=["base", "large-self"])
args = parser.parse_args()

MODEL_BASE = "wav2vec2-base-960h-4-gram"
MODEL_LARGE_SELF = "wav2vec2-large-960h-lv60-self-4-gram"

if args.model == "base":
    MODEL_NAME = MODEL_BASE
elif args.model == "large-self":
    MODEL_NAME = MODEL_LARGE_SELF

THRESHOLD, THRESHOLD_VALUE = None, 0.75
PREDICTIONS_NAME = f"filtered_bs_thresholded_{THRESHOLD_VALUE}_nosamewords" if THRESHOLD else "filtered_bs_argmin_nosamewords"

ls_test_other = datasets.Dataset.load_from_disk(f"{DATA_PATH}predictions/{MODEL_NAME}-test_other")
ls_test_clean = datasets.Dataset.load_from_disk(f"{DATA_PATH}predictions/{MODEL_NAME}-test_clean")
ls_test_all = datasets.concatenate_datasets([ls_test_clean, ls_test_other])

filtered_beam_search_other = datasets.Dataset.load_from_disk(f"{DATA_PATH}predictions/{MODEL_NAME}-{PREDICTIONS_NAME}-test_other")
filtered_beam_search_clean = datasets.Dataset.load_from_disk(f"{DATA_PATH}predictions/{MODEL_NAME}-{PREDICTIONS_NAME}-test_clean")
filtered_beam_search = datasets.concatenate_datasets([filtered_beam_search_clean, filtered_beam_search_other])

wrong_transcriptions_cnt, first_is_most_probable_both_cnt, wrong_transcription_changed_after_bsfiltering_cnt, correct_transcription_changed_after_bsfiltering_cnt, cnt, cnt2 = 0, 0, 0, 0, 0, 0
new_transcription_corrected_cnt = 0
top = Counter()
for i in range(len(ls_test_all)):
    text = ls_test_all[i]["text"]
    transcription = ls_test_all[i]["transcription"]
    candidates = ls_test_all[i]["candidates"]
    new_transcription = filtered_beam_search[i]["transcription"]
    new_candidates = filtered_beam_search[i]["candidates"]
    if transcription != text:
        wrong_transcriptions_cnt += 1
        
        if len(ls_test_all[i]["candidates"]) > 1:
            p = [vv for vv in zip(ls_test_all[i]["lm_probability"], ls_test_all[i]["logit_probability"])]
            max_probability_indices = torch.argmax(torch.tensor([[x[ii] for x in p] for ii in range(len(p[0]))]), dim=-1)
            if torch.all(torch.tensor([x == max_probability_indices[0] for x in max_probability_indices])):
                first_is_most_probable_both_cnt += 1
                
        if transcription != new_transcription:
            wrong_transcription_changed_after_bsfiltering_cnt += 1
            if text == new_transcription:
                new_transcription_corrected_cnt += 1
        
        if text in candidates:
            cnt += 1
            for ii, candidate in enumerate(candidates):
                if text == candidate:
                    top[ii] += 1
        elif text in new_candidates:
            cnt2 += 1
            for ii, candidate in enumerate(new_candidates):
                if text == candidate:
                    top[ii] += 1
    elif transcription != new_transcription:
            correct_transcription_changed_after_bsfiltering_cnt += 1
        

print("Model", MODEL_NAME)
print(f"Total n. of wrong transcriptions: {wrong_transcriptions_cnt}/{len(ls_test_all)} ({round(wrong_transcriptions_cnt / len(ls_test_all) * 100, 2)}%)")
print(f"Picked (wrong) transcription is the most probable for both LM and ASR models {first_is_most_probable_both_cnt}/{wrong_transcriptions_cnt} times ({round(first_is_most_probable_both_cnt / wrong_transcriptions_cnt * 100, 2)}%)")
print(f"Correct transcription is in LM candidates {cnt}/{wrong_transcriptions_cnt} times ({round(cnt / wrong_transcriptions_cnt * 100, 2)}%)")
print(f"Picked (wrong) transcription changed {wrong_transcription_changed_after_bsfiltering_cnt} times after Beam Search filtering ({new_transcription_corrected_cnt} are now correct)")
print(f"Picked (correct) transcription changed {correct_transcription_changed_after_bsfiltering_cnt} times after Beam Search filtering")
print("Correct transcription added among LM candidates", cnt2, "times after Beam Search filtering")
print("Correct transcription is among the next top-3 candidates", top[1] + top[2] + top[3], "times")
print("Correct transcription is among the next top-5 candidates", top[1] + top[2] + top[3] + top[4] + top[5], "times")