import streamlit as st

s1 = st.text_input("Sense IDs sentence #0", "bn:00026300n bn:00066514n bn:00088629v bn:00066514n bn:00041942n")

s2 = st.text_input("Sense IDs sentence #1 (last id is missing)", "bn:00026300n bn:00066514n bn:00088629v bn:00066514n")

s3 = st.text_input("Sense IDs sentence #2 (wrong id at idx 2)", "bn:00026300n bn:00066514n bn:00093172v bn:00066514n bn:00041942n")

s4 = st.text_input("Sense IDs sentence #3 (id at idx 2 is missing)", "bn:00026300n bn:00066514n bn:00066514n bn:00041942n")

s5 = st.text_input("Sense IDs sentence #4 (new id appended)", "bn:00026300n bn:00066514n bn:00088629v bn:00066514n bn:00041942n bn:00088629v")

ll = [x.split(" ") for x in [s1, s2, s3, s4, s5]]

import kenlm
import torch
model = kenlm.LanguageModel("../models/ngrams/4gram.binary")

len_mean = lambda liste : torch.tensor([len(lista) for lista in liste]).float().mean()
get_weights = lambda max: [(i+1 if i+1 < 4 else 4) for i in range(max)]
get_weights_reversed = lambda max: [(4-i if 4-i > 1 else 1) for i in range(max)]
compute_scores 	 = lambda lista : [model.score(" ".join(lista[:i+1])) for i in range(len(lista))]

aggregations_sum = [torch.tensor(compute_scores(lista)).sum() for lista in ll]
aggregations = [torch.tensor(compute_scores(lista)).mean() for lista in ll]
len_mean_computed = len_mean([compute_scores(lista) for lista in ll])
aggregations_2 = [(torch.tensor(compute_scores(lista)) / len_mean_computed).mean() for lista in ll]
aggregations_3 = [(torch.tensor(compute_scores(lista)) * torch.tensor(get_weights(len(compute_scores(lista))))).mean() for lista in ll]
aggregations_4 = [(torch.tensor(compute_scores(lista)) * torch.tensor(get_weights_reversed(len(compute_scores(lista))))).mean() for lista in ll]
distribution_sum = torch.tensor(aggregations_sum).softmax(dim=0)
distribution = torch.tensor(aggregations).softmax(dim=0)
distribution_2 = torch.tensor(aggregations_2).softmax(dim=0)
distribution_3 = torch.tensor(aggregations_3).softmax(dim=0)
distribution_4 = torch.tensor(aggregations_4).softmax(dim=0)
# asr_probabilities = torch.tensor([x["logit_probability"] for lista in ll])
# st.write(f"**ASR probabilities**: (argmax #{asr_probabilities.argmax(-1)})", asr_probabilities)
# lm_probabilities = torch.tensor([x["logit_probability"] for lista in ll])
# st.write(f"**LM probabilities**: (argmax #{lm_probabilities.argmax(-1)})", lm_probabilities)
st.markdown(f"**WSD LM probabilities**: \n"
	+ f"- Sum (argmax #{distribution_sum.argmax(-1)}) `{distribution_sum}` \n"
	+ f"- Average (argmax #{distribution.argmax(-1)}) `{distribution}` \n"
	+ f"- Average w/ averaged lengths (argmax #{distribution_2.argmax(-1)}) `{distribution_2}` \n"
	+ f"- Weighted average (argmax #{distribution_3.argmax(-1)}) `{distribution_3}` \n"
	+ f"- Reversed weighted average (argmax #{distribution_4.argmax(-1)}) `{distribution_4}`")

st.write("**Scores**:", [compute_scores(lista) for lista in ll])
