
# 형태소만 뽑아 저장
stopwords = ['은','는','가','하','아','것','들','의','되','수','보','주','등','한']

def tense_tokenizer(sentence, pos=["J","E"], stopword=stopwords): 

    from konlpy.tag import Hannanum
    hannanum = Hannanum()

    sentence = [word for word, tag in hannanum.pos(sentence) if len(word) > 0 and tag in pos and word not in stopword]

    return sentence


# tfidf, model
def tense_ml(sentence):

    import joblib

    loaded_vectorizer = joblib.load('./tense_vectorizer_customized.pkl') 
    loaded_model = joblib.load('./tense_model_customized.pkl') 

    #df_token_list = [str(x).lower() for x in df]
    df_token = tense_tokenizer(sentence)       
    df_tfidf = loaded_vectorizer.transform(df_token)

    pred = loaded_model.predict(df_tfidf)
    pred = pred[-1]

    return pred

if __name__ == "__main__":
    print(tense_ml('김금희 소설가는 ＂계약서 조정이 그리 어려운가 작가를 격려한다면서 그런 문구 하나 고치기가 어려운가 작가의 노고와 권리를 존중해줄 수 있는 것 아닌가＂라고 꼬집었다.'))
