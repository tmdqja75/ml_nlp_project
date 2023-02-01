import numpy as np
import re

from joblib import load

from konlpy.tag import Mecab

STOPWORDS = ['은', '는', '도', '한', '이다', '을', '이', '를', '가', '에', '의', '과', '에서', '으로', '들', '로', '와', '등']


def custom_tokenizer(sentence):
    '''
    각 문장을 Mecab을 이용하여 토큰화해줄 함수
    토큰들을 리스트 형식으로 반환
    '''
    t= Mecab()
    return [token[0] for token in t.pos(sentence)]

# regex 함수
def regex_filter(sentence):
    return re.sub(r"[^가-힣\s!?]|\(.*?\)", "", sentence)

def type_ml_model(sent):
    vectorizer = load('final_vectorizer.joblib')
    model = load('final_model.joblib')

    sent_regex_filter = regex_filter(sent)
    sent_tfidf = vectorizer.transform([sent_regex_filter])

    prediction = model.predict(sent_tfidf)
    # return model.predict(sent_tfidf)
    return prediction


if __name__ == "__main__":
    print(type_ml_model('''"장욱진의 ＇가족＇은 허물 없는 가족애를, 처음 공개되는 정약용의 ＇정효자전＇과 ＇정부인전＇은 강진 사람 정여주의 부탁을 받아 그의 일찍 죽은 아들과 홀로 남은 며느리의 안타까운 사연을 쓴 서예 작품이다."'''))


