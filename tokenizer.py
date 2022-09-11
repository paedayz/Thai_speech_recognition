import __main__
from pythainlp.tokenize import dict_trie, dict_word_tokenize
from pythainlp.corpus.common import thai_words

def tokenize(text):
    custom_words_list = set(thai_words())
    custom_words_list.update(__main__.important_words)
    trie = dict_trie(dict_source=custom_words_list)

    result = dict_word_tokenize(text, custom_dict=trie)

    return result