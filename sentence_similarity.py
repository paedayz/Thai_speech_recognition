from pythainlp import word_tokenize # ทำการเรียกตัวตัดคำ
from sklearn.metrics.pairwise import cosine_similarity  # ใช้หาค่าความคล้ายคลึง
import numpy as np
from pythainlp.corpus import download
from pythainlp.corpus import get_corpus_path


model=download('thai2fit_wv') # ดึง model ของ thai2vec มาเก็บไว้ในตัวแปร model
def sentence_vectorizer(ss,dim=300,use_mean=True): # ประกาศฟังก์ชัน sentence_vectorizer
    s = word_tokenize(ss)
    vec = np.zeros((1,dim))
    for word in s:
        if word in model.wv.index2word:
            vec+= model.wv.word_vec(word)
        else: pass
    if use_mean: vec /= len(s)
    return vec
def sentence_similarity(s1,s2):
    return cosine_similarity(sentence_vectorizer(str(s1)),sentence_vectorizer(str(s2)))

print(sentence_similarity("ผมเป็นนักศึกษาเรียนที่มหาวิทยาลัยขอนแก่น","ผมเป็นนักศึกษามหาวิทยาลัยขอนแก่น"))