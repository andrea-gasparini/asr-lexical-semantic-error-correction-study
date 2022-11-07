import os
from typing import Dict, Tuple, List, Optional, Set, Literal

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, AutoModelForCTC, AutoProcessor
from transformers.modeling_outputs import CausalLMOutput
from pyctcdecode import BeamSearchDecoderCTC

from constants import MODELS_PATH
from utils.metrics import PointwiseMutualInformation as PMI
from utils import list_to_dict


def load_pretrained_model(hf_model_name: str, local_dumps_path: str = MODELS_PATH) -> Tuple[AutoModelForCTC, AutoProcessor]:
    """
    Loads an Hugging Face pretrained model from the hub and saves a local dump of it to the specified directory.
    When a model has previosuly been dumped locally, the function directly loads it without any downloads from the hub.

    Args:
        hf_model_name (`str`):
            Model's name in Hugging Face, e.g. "facebook/wav2vec2-base-960h".
        local_dumps_path (`str`, optional, defaults to `MODELS_PATH`)
            Path to the directory in which save the model's dump.

    Returns:
        `Tuple[AutoModelForCTC, AutoProcessor]`:
            A pair of pretrained model and processor.
    """
    local_dump_path = os.path.join(local_dumps_path, hf_model_name)

    if not os.path.isdir(local_dump_path):
        model = AutoModelForCTC.from_pretrained(hf_model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(hf_model_name)
        model.save_pretrained(local_dump_path)
        processor.save_pretrained(local_dump_path)
    else:
        model = AutoModelForCTC.from_pretrained(local_dump_path).to("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(local_dump_path)

    return model, processor


class Wav2Vec2:

    def __init__(self, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor) -> None:
        super().__init__()

        if not isinstance(processor, Wav2Vec2Processor):
            raise ValueError(f"`processor` must be of type {Wav2Vec2Processor}, but is {type(processor)}")
        
        if not isinstance(model, Wav2Vec2ForCTC):
            raise ValueError(f"`model` must be of type {Wav2Vec2ForCTC}, but is {type(model)}")
        
        self.model, self.processor = model, processor
    
    @classmethod
    def from_pretrained(cls, hf_model_name: str):
        model, processor = load_pretrained_model(hf_model_name)
        return cls(model, processor)

    def predict(self, batch: Dict) -> Dict:
        return self.greedy_search(batch)

    def forward(self, batch: Dict) -> CausalLMOutput:
        if batch["audio"]["sampling_rate"] != self.processor.feature_extractor.sampling_rate:
            raise ValueError("given batch sampling rate does not match the processor's one. Must be resampled!")

        inputs = self.processor(batch["audio"]["array"], return_tensors="pt", padding="longest",
                                sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs)

        return out

    def greedy_search(self, batch: Dict) -> Dict:
        logits = self.forward(batch).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        batch["transcription"] = transcription
        return batch


class Wav2Vec2WithLM(Wav2Vec2):

    def __init__(self, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor, beam_decoder_path: str) -> None:
        super().__init__(model, processor)
        self.beam_decoder = BeamSearchDecoderCTC.load_from_dir(beam_decoder_path)
        
    @classmethod
    def from_pretrained(cls, hf_model_name: str, beam_decoder_path: str):
        model, processor = load_pretrained_model(hf_model_name)
        return cls(model, processor, beam_decoder_path)

    def predict(self, batch: Dict) -> Dict:
        return self.beam_search(batch)

    def beam_search(self, batch: Dict) -> Dict:
        logits = self.forward(batch).logits

        output_beams = self.beam_decoder.decode_beams_batch(None, logits.cpu().numpy())[0]

        lm_scores = [x[-1] for x in output_beams]
        logit_scores = [x[-2] for x in output_beams]
        batch["candidates"] = [x[0] for x in output_beams]
        batch["transcription"] = batch["candidates"][0]

        batch["lm_probability"] = torch.softmax(torch.tensor(lm_scores), dim=-1)
        batch["logit_probability"] = torch.softmax(torch.tensor(logit_scores), dim=-1)

        return batch


class Wav2Vec2WithWSD(Wav2Vec2WithLM):

    def __init__(self, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor, beam_decoder_path: str,
                 wsd_tagged_predictions: Dict[str, List[Dict]], include_most_probable_candidates: bool = True,
                 threshold: Optional[float] = None, argmin: bool = True, pmi: Optional[PMI] = None,
                 pmi_mode: Literal["average", "1-vs-all"] = "average") -> None:
        super().__init__(model, processor, beam_decoder_path)
        self.pmi = pmi
        self.pmi_mode = pmi_mode
        self.argmin = argmin
        self.threshold = threshold
        self.wsd_tagged_predictions = wsd_tagged_predictions
        self.sample_ids_to_filter = self.__compute_sample_ids_to_filter(include_most_probable_candidates)
        
    def __compute_sample_ids_to_filter(self, most_probable_candidates: bool) -> Set[str]:
        """
        Computes the sample ids for which the beam search will have to look for possible beams to filter.

        Args:
            most_probable_candidates (`bool`):
                Whether to include samples with the chosen transcription being the most probable
                among the other candidates for both the LM and the ASR model.
        """
        sample_ids_to_filter = set()
    
        for sample_id, sample_candidates in self.wsd_tagged_predictions.items():

            # skip samples with only one candidate transcription
            if len(sample_candidates) == 1: 
                continue

            # skip samples with the chosen transcription being the most probable for both LM and ASR models
            if not most_probable_candidates:
                lm_probabilities = [float(candidate["lm_probability"]) for candidate in sample_candidates]
                asr_probabilities = [float(candidate["logit_probability"]) for candidate in sample_candidates]
                max_probability_indices = torch.argmax(torch.tensor([lm_probabilities, asr_probabilities]), dim=-1)
                if max_probability_indices[0] == 0 and max_probability_indices[1] == 0:
                    continue

            # take into consideration only samples with at least one candidate transcription
            # containing different BabelNet identifiers with respect to the other candidates
            predictions = [candidate["bn_esc_predictions"] for candidate in sample_candidates]
            if not torch.all(torch.tensor([pred == predictions[0] for pred in predictions])):
                sample_ids_to_filter.add(sample_id)
        
        return sample_ids_to_filter
        
    def predict(self, sample: Dict) -> Dict:
        return self.filtered_beam_search(sample)

    def filtered_beam_search(self, sample: Dict) -> Dict:
        beams_to_filter = list()
        candidates = list_to_dict(self.wsd_tagged_predictions[sample["id"]])

        if sample["id"] in self.sample_ids_to_filter:
            if self.pmi is not None:
                if self.pmi_mode == "average":
                    beams_to_filter = self.compute_beams_to_filter(candidates)
                else:
                    beams_to_filter = self.compute_beams_to_filter_pmi_1_vs_all(candidates)
            else:
                beams_to_filter = self.compute_beams_to_filter(candidates)
        # use the transcription already computed by the previous beam search, if available
        elif "transcription" in candidates:
            sample["candidates"] = candidates["transcription"]
            sample["transcription"] = sample["candidates"][0]
            return sample

        logits = self.forward(sample).logits[0]
        output_beams = self.beam_decoder.decode_beams(logits.cpu().numpy(), beams_to_filter=beams_to_filter)
        sample["candidates"] = [x[0] for x in output_beams]
        sample["transcription"] = sample["candidates"][0] if len(sample["candidates"]) > 0 else ""

        return sample

    def compute_beams_to_filter(self, sample: Dict[str, List]) -> List[str]:
        tokens = sample["tokens"]
        indices = sample["sense_indices"]
        bn_predictions = sample["bn_esc_predictions"]
        scores = sample["wsd_lm_scores" if self.pmi is None else "wsd_pmi_scores"]        

        # TODO: we're not considering sequences with less senses before `i` than the 1st one

        beams_to_filter = list()

        for i, transcription_bn_id in enumerate(bn_predictions[0]):
            
            if self.pmi is not None and transcription_bn_id not in self.pmi.unigram_frequences:
                continue
            
            tran_bn_id_idx = indices[0][i]
            tran_tokens = tokens[0]

            tmp = [i]
            bn_ids_window = [transcription_bn_id]
            scores_window = [scores[0][i]]
            indices_window = [indices[0]]
            tokens_window = [tokens[0]]

            for ii, candidate_bn_ids in enumerate(bn_predictions):

                # skip transcription candidate (already inserted)
                if ii == 0:
                    continue

                if len(candidate_bn_ids) > i or self.pmi is not None:

                    cand_tokens = tokens[ii]

                    if self.pmi is None:
                        cand_sense_idx = i
                        cand_bn_id_idx = indices[ii][cand_sense_idx]
                    else:
                        bn_id_idx_pairs = ((idx, j) for j, idx in enumerate(indices[ii]) if idx == tran_bn_id_idx)
                        cand_bn_id_idx, cand_sense_idx = next(bn_id_idx_pairs, (None, None))      

                    if cand_bn_id_idx is None or self.pmi is not None and candidate_bn_ids[cand_sense_idx] not in self.pmi:
                        continue

                    tokens_are_different = tran_tokens[tran_bn_id_idx] != cand_tokens[cand_bn_id_idx]
                    bn_ids_are_equal = transcription_bn_id == candidate_bn_ids[cand_sense_idx]

                    if tran_bn_id_idx == cand_bn_id_idx and (tokens_are_different or bn_ids_are_equal):
                        tmp.append(cand_sense_idx)
                        bn_ids_window.append(candidate_bn_ids[cand_sense_idx])
                        scores_window.append(scores[ii][cand_sense_idx])
                        indices_window.append(indices[ii])
                        tokens_window.append(tokens[ii])

            if any([id for id in bn_ids_window if id != bn_ids_window[0]]):
                if self.threshold is None:
                    if self.argmin:
                        min_idx = torch.argmin(torch.tensor(scores_window))
                        beams_to_filter.append(" ".join(tokens_window[min_idx][:indices_window[min_idx][tmp[min_idx]] + 1]))
                    else:
                        min_value = min(scores_window)
                        for el_index, el in enumerate(scores_window):
                            if el == min_value:
                                beams_to_filter.append(" ".join(tokens_window[el_index][:indices_window[el_index][tmp[min_idx]] + 1]))
                else:
                    max_value = max(scores_window)
                    for el_index, el in enumerate(scores_window):
                        delta = max_value - el
                        if delta > self.threshold:
                            beams_to_filter.append(" ".join(tokens_window[el_index][:indices_window[el_index][tmp[min_idx]] + 1]))

        return beams_to_filter

    def compute_beams_to_filter_pmi_1_vs_all(self, sample: Dict[str, List]) -> List[str]:
        threshold = self.threshold if self.threshold is not None else 0.0
        senses_lists = sample["bn_esc_predictions"]
        sense_indices = sample["sense_indices"]
        tokens = sample["tokens"]

        beams_to_filter = list()

        for candidate_idx, senses in enumerate(senses_lists):

            senses = [sense for sense in senses if sense in self.pmi.unigram_frequences]

            for sense_idx, sense in enumerate(senses):
                is_valid, is_in_pmi = False, False
                for sense_idx2, sense2 in enumerate(senses):
                    
                    if sense_idx == sense_idx2:
                        continue
                    
                    if f"{sense} {sense2}" in self.pmi.bigram_frequences:
                        is_in_pmi = True
                        v = self.pmi.pmi(sense, sense2)				
                        if v > threshold:
                            is_valid = True
                            break
                    else:
                        # TODO can we do better than ignoring? (case w/ unseen pair in the train corpus)
                        continue
                        # is_in_pmi = True

                if not is_valid:				
                    if is_in_pmi:
                        sense_token_index = sense_indices[candidate_idx][sense_idx]
                        beams_to_filter.append(" ".join(tokens[candidate_idx][:sense_token_index + 1]))
            
        return beams_to_filter
