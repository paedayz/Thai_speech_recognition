import argparse
from speech_recog_wav2vec2_th import Wav2Vec2_TH
from speech_recog_google import SpeechRecogGoogle
from word_frequency import generate_word_frequency_report
import pandas as pd
from config import important_words
from config import stop_words

def main(args):
    if args.recog == 'google':
        speechRecogGoogle = SpeechRecogGoogle()
        audio_to_text = speechRecogGoogle.speech_recognition('voice/my_voice.wav')
    else:
        wav2vec2_th = Wav2Vec2_TH()
        audio_to_text = wav2vec2_th.speech_recognition('voice/my_voice.wav')

    print('text from recognition:')
    print(audio_to_text)

    generate_word_frequency_report(audio_to_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--recog",
        type=str,
        default="wav2vec2_th",
        required=False
    )

    args = parser.parse_args()

    main(args)

