import speech_recognition as sr

class SpeechRecogGoogle:
    def __init__(self) -> None:
        self.recog = sr.Recognizer()

    def speech_recognition(self, voicePath) -> str:
        audio_file = sr.AudioFile(voicePath)

        with audio_file as source: 
            self.recog.adjust_for_ambient_noise(source) 
            audio = self.recog.record(source)

        print('google recognizing ...')
        text = self.recog.recognize_google(audio, language='th')
        return text