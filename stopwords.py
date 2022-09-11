from pythainlp.tokenize import word_tokenize
text = "มีการความทดสอบภาษาไทย"
list_word = word_tokenize(text)
print(list_word)

from pythainlp.corpus import thai_stopwords
stopwords = list(thai_stopwords())
list_word_not_stopwords = [i for i in list_word if i not in stopwords]
print(list_word_not_stopwords)