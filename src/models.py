from typing import Dict

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers.modeling_outputs import CausalLMOutput

from pyctcdecode import BeamSearchDecoderCTC


class Wav2Vec2:

    def __init__(self, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor) -> None:
        super().__init__()

        if not isinstance(processor, Wav2Vec2Processor):
            raise ValueError(f"`processor` must be of type {Wav2Vec2Processor}, but is {type(processor)}")
        
        if not isinstance(model, Wav2Vec2ForCTC):
            raise ValueError(f"`model` must be of type {Wav2Vec2ForCTC}, but is {type(model)}")
        
        self.model, self.processor = model, processor

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
