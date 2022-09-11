import speech_recognition as sr
import pandas as pd

from pythainlp.tokenize import dict_trie, dict_word_tokenize
from pythainlp.corpus.common import thai_words
from speech_to_text import speech_recognition

def speech_recog_function():
    recog = sr.Recognizer()
    audio_file = sr.AudioFile('voice/pp.wav')

    with audio_file as source: 
        recog.adjust_for_ambient_noise(source) 
        audio = recog.record(source)
   
    print('recognizing now loading ...')
    text = recog.recognize_google(audio, language="th", show_all=True)
    return text

word_df = pd.read_csv('Importantandstopwords.csv', index_col=False)

important_words = word_df[word_df['Is not important'] == 1]['Word'].values
stop_words = word_df[word_df['Is not important'] == 0]['Word'].values

audio_to_text = speech_recognition()
audio_to_text = audio_to_text.replace(" ", "")

# Tokenizer
custom_words_list = set(thai_words())
custom_words_list.update(important_words)
trie = dict_trie(dict_source=custom_words_list)

text_list = dict_word_tokenize(audio_to_text, custom_dict=trie)


filter_important_text = [i for i in text_list if i in important_words]

print(important_words)

filter_out_stop_word_text = [i for i in text_list if i not in stop_words and i not in filter_important_text and len(i) > 1]

text_dict = {}
for word in filter_important_text + filter_out_stop_word_text:
    if word not in text_dict:
        text_dict[word] = audio_to_text.count(word)

sort_dict = dict(reversed(sorted(text_dict.items(), key=lambda item: item[1])))

result_df = pd.DataFrame(sort_dict.items(), columns=['word', 'frequency'])
result_df.to_csv('report.csv', encoding='utf-8')
print(result_df)

print(audio_to_text)
print("คอสออฟคาร์บอน" in audio_to_text)
print(text_list)


print('\nsave to report.csv successfully')

