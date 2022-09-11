import torchaudio
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
#load pretrained processor and model
processor = Wav2Vec2Processor.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
model = Wav2Vec2ForCTC.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")

#function to resample to 16_000
def speech_file_to_array_fn(batch, 
                            text_col="sentence", 
                            fname_col="path",
                            resampling_to=16000):
    speech_array, sampling_rate = torchaudio.load(batch[fname_col])
    resampler=torchaudio.transforms.Resample(sampling_rate, resampling_to)
    batch["speech"] = resampler(speech_array)[0].numpy()
    batch["sampling_rate"] = resampling_to
    batch["target_text"] = batch[text_col]
    return batch

def speech_recognition():
    speech, rate = librosa.load("voice/42_Khlong Nueng.wav",sr=16000)
    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values,).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    print("Prediction:", processor.batch_decode(predicted_ids))
    return processor.batch_decode(predicted_ids)[0]