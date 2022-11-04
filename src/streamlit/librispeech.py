from faulthandler import disable
import streamlit as st
import json
from constants import *

with open(f"{DATA_PATH}/librispeech/librispeech_test_all_scored.json") as f:
	samples = json.load(f)



sure_samples = dict()
samples_with_candidates = dict()

for k, v in samples.items():
	if len(v) == 1:
		sure_samples[k] = v[0]
	else:
		samples_with_candidates[k] = v



import torch

uncertain_samples = dict()

for k, v in samples_with_candidates.items():
	p = [(float(vv["lm_probability"]), float(vv["logit_probability"])) for vv in v]
	max_probability_indices = torch.argmax(torch.tensor([[x[i] for x in p] for i in range(len(p[0]))]), dim=-1)
	if torch.all(torch.tensor([x == max_probability_indices[0] for x in max_probability_indices])):
		continue
	else:
		uncertain_samples[k] = v



wsd_differences_samples = dict()

for k, v in uncertain_samples.items():
	predictions = [vv["esc_predictions"] for vv in v]
	if not torch.all(torch.tensor([e == predictions[0] for e in predictions])):
		wsd_differences_samples[k] = v



wsd_differences_substitution_samples = dict()

for i, (k, v) in enumerate(wsd_differences_samples.items()):
	sense_ids = [vv["sense_indices"] for vv in v]
	if torch.all(torch.tensor([e == sense_ids[0] for e in sense_ids])):
		wsd_differences_substitution_samples[k] = v



tot = len(wsd_differences_samples.values()) # len(wsd_differences_samples.values())

sample_index = st.number_input("Sample index:", value=778, min_value=0, max_value=tot-1)

k, v = list(wsd_differences_samples.items())[sample_index]

import datasets

ls_test_other = datasets.Dataset.load_from_disk(f"{DATA_PATH}librispeech/librispeech_test_other")
ls_test_clean = datasets.Dataset.load_from_disk(f"{DATA_PATH}librispeech/librispeech_test_clean")
ls_test_all = datasets.concatenate_datasets([ls_test_clean, ls_test_other])

original_texts = {v["id"]: v["text"] for v in ls_test_all}

len_mean = lambda liste : torch.tensor([len(lista) for lista in liste]).float().mean()
get_weights = lambda max: [(i+1 if i+1 < 4 else 4) for i in range(max)]
get_weights_reversed = lambda max: [(4-i if 4-i > 1 else 1) for i in range(max)]

st.markdown(f"**LibriSpeech sentence {k}**: \n - {original_texts[k].lower()}")

st.write("**Transcription candidates**: \n")

for i, vv in enumerate(v):
	# url = f"https://babelnet.org/synset?id=bn%3A{offset_pos}&orig=bn%3A{offset_pos}&lang=EN"
	tokens_underlined_senses = ["<u>" + e + "</u>" if ii in vv["sense_indices"] else e for ii, e in enumerate(vv["tokens"])]
	st.markdown(f"{i}) " + " ".join(tokens_underlined_senses), unsafe_allow_html=True)

aggregations_sum = [torch.tensor(x["wsd_lm_scores"]).sum() for x in v]
aggregations = [torch.tensor(x["wsd_lm_scores"]).mean() for x in v]
len_mean_computed = len_mean([x["wsd_lm_scores"] for x in v])
aggregations_2 = [(torch.tensor(x["wsd_lm_scores"]) / len_mean_computed).mean() for x in v]
aggregations_3 = [(torch.tensor(x["wsd_lm_scores"]) * torch.tensor(get_weights(len(x["wsd_lm_scores"])))).mean() for x in v]
aggregations_4 = [(torch.tensor(x["wsd_lm_scores"]) * torch.tensor(get_weights_reversed(len(x["wsd_lm_scores"])))).mean() for x in v]

distribution_sum = torch.tensor(aggregations_sum).softmax(dim=0)
distribution = torch.tensor(aggregations).softmax(dim=0)
distribution_2 = torch.tensor(aggregations_2).softmax(dim=0)
distribution_3 = torch.tensor(aggregations_3).softmax(dim=0)
distribution_4 = torch.tensor(aggregations_4).softmax(dim=0)

asr_probabilities = torch.tensor([x["logit_probability"] for x in v])
st.write(f"**ASR probabilities**: (argmax #{asr_probabilities.argmax(-1)})", asr_probabilities)
lm_probabilities = torch.tensor([x["lm_probability"] for x in v])
st.write(f"**LM probabilities**: (argmax #{lm_probabilities.argmax(-1)})", lm_probabilities)
st.markdown(f"**WSD LM probabilities**: \n"
	+ f"- Sum (argmax #{distribution_sum.argmax(-1)}) `{distribution_sum}` \n"
	+ f"- Average (argmax #{distribution.argmax(-1)}) `{distribution}` \n"
	+ f"- Average w/ averaged lengths (argmax #{distribution_2.argmax(-1)}) `{distribution_2}` \n"
	+ f"- Weighted average (argmax #{distribution_3.argmax(-1)}) `{distribution_3}` \n"
	+ f"- Reversed weighted average (argmax #{distribution_4.argmax(-1)}) `{distribution_4}`")

col1, col2 = st.columns(2)

with col1:
	st.write("**WSD LM scores**:", [x["wsd_lm_scores"] for x in v])

with col2:
	st.write("**BN IDs**:", [x["bn_esc_predictions"] for x in v])