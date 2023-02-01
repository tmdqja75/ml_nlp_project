def apply_re(text):
    import re

    text = text.replace('.', ' ')
    hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')
    result = hangul.sub('', text)
    return result

def last_two_word(sentence):
    result = sentence.split(" ")[-2:]
    return result[0] + ' ' + result[1]

def data_preprocessing(raw_data):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd

    raw_data_cer_sample = raw_data[raw_data["확실성"] == "확실"].sample(3000, random_state=13)
    raw_data_un_sample = raw_data[raw_data["확실성"] == "불확실"]
    raw_data_sample = pd.concat([raw_data_cer_sample, raw_data_un_sample])
    
    le = LabelEncoder()
    raw_data_sample["y"] = le.fit_transform(raw_data_sample["확실성"])

    vectorizer = TfidfVectorizer(min_df=1, decode_error='ignore')
    X = vectorizer.fit_transform(raw_data_sample["문장"].apply(last_two_word).apply(apply_re).values)
    y = raw_data_sample["y"].values

    return X, y, vectorizer

def data_split(X, y):
    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=13)

    return X_train, X_test, y_train, y_test

def fit_model(raw_data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd

    X, y, vectorizer = data_preprocessing(raw_data)
    X_train, X_test, y_train, y_test = data_split(X, y)

    params = {'C':[0.01, 0.1,1,5,10], 'max_iter': [200,500,1000]}

    lr = LogisticRegression(random_state=13)
    skfold = StratifiedKFold(n_splits=5)
    grid = GridSearchCV(lr, params,scoring='roc_auc' , cv=skfold)

    grid.fit(X_train, y_train)

    return vectorizer, grid.best_estimator_

def predict(sentence, vectorizer, model):
    sentence_pre = apply_re(last_two_word(sentence))
    sentence_vec = vectorizer.transform([sentence_pre])

    predict_answer = model.predict(sentence_vec)

    if predict_answer == 0:
        return "불확실"
    elif predict_answer == 1:
        return "확실"

'''
import sentence_classify as cls
import pandas as pd

raw_data = pd.read_csv('../ML_project/data/train.csv')

sentence = '내 삶은 아직 끝나지 않을 예정이다.'
vectorizer, model = cls.fit_model(raw_data)
cls.predict(sentence, vectorizer, model)
'''