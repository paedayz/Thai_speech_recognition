import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class Wav2Vec2_TH:
    def __init__(self) -> None:
        self.processor = Wav2Vec2Processor.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
        self.model = Wav2Vec2ForCTC.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")

    def speech_recognition(self, voice_path: str):
        speech, rate = librosa.load(voice_path, sr=16000)
        inputs = self.processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)

        print('Wav2Vec2_TH recognizing ...')
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)

        return self.processor.batch_decode(predicted_ids)[0].replace(" ", "")