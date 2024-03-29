import argparse
from pydub import AudioSegment
import math

class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '\\' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '\\' + split_filename, format="wav")
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 60)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min_per_split",
        type=str,
        default="2",
        required=False
    )

    args = parser.parse_args()

    folder = 'voice'
    file = 'Recording.wav'
    splitWavAudioMubin = SplitWavAudioMubin(folder, file)
    splitWavAudioMubin.multiple_split(min_per_split=int(args.min_per_split))

