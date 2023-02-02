import pandas as pd
import re

from konlpy.tag import Okt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
import os 



def text_cleaning(text):
  # 불용어 사전 불러오기
  stopwords = pd.read_csv(
    "https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/korean_stopwords.txt"
  ).values.tolist()

  custom_stopwords = [
    '의', '가', '이', '은', '들', '는', '좀', '과', '도', '를', '을', '으로', '자', '에', '와',
    '것', '등', '년', '월', '일', '수', '그', '중', '명', '때', '및', '했다', '에서', '로',
    '이다', '까지', '에는'
  ]

  hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')  # 정규 표현식 처리 한글이 아닌 글자는 제거
  result = hangul.sub('', str(text))
  okt = Okt()
  morphs = okt.morphs(result)  # 형태소 추출
  morphs = [x for x in morphs if x not in custom_stopwords]  # 커스텀 불용어 제거
  morphs = [x for x in morphs if x not in stopwords]  # 불용어 제거
  return morphs


def getDict(df):

  try:
    pos = pd.read_csv('pos_dict.csv')['pos'].values
    nega = pd.read_csv('nega_dict.csv')['nega'].values
    neu = pd.read_csv('neu_dict.csv')['neu'].values

  except:
    corpus = "".join(df[df['극성'] == '긍정']['문장'].tolist())
    pos = (text_cleaning(corpus))

    corpus = "".join(df[df['극성'] == '부정']['문장'].tolist())
    nega = (text_cleaning(corpus))

    corpus = "".join(df[df['극성'] == '미정']['문장'].tolist())
    neu = (text_cleaning(corpus))

    pos_dict = pd.DataFrame({
      'pos': pos,
    })
    nega_dict = pd.DataFrame({
      'nega': nega,
    })
    neu_dict = pd.DataFrame({
      'neu': neu,
    })

    pos_dict.to_csv('pos_dict.csv')
    nega_dict.to_csv('nega_dict.csv')
    neu_dict.to_csv('neu_dict.csv')

  return pos, nega, neu


# 주어진 문장의 단어들을 극성별로 사전에 등록되어 있는 갯수 카운팅
def getX(df):
  pos_cnt = []
  nega_cnt = []
  neu_cnt = []

  pos_dict, nega_dict, neu_dict = getDict(df)

  try:
    path = os.getcwd() + '/type_sh/cnt_df.csv'
    cnt_df = pd.read_csv(path, index_col=0)

  except:
    for idx, row in df.iterrows():
      corpus = "".join(row['문장'])
      nouns = text_cleaning(corpus)

      cnt1 = 0
      cnt2 = 0
      cnt3 = 0

      for word in nouns:
        if word in pos_dict:
          cnt1 += 1
        if word in nega_dict:
          cnt2 += 1
        if word in neu_dict:
          cnt3 += 1

      pos_cnt.append(cnt1), nega_cnt.append(cnt2), neu_cnt.append(cnt3)

    cnt_df = pd.DataFrame({'pos': pos_cnt, 'nega': nega_cnt, 'neu': neu_cnt})

  return cnt_df


def transformTarget(s, df):
  pos_cnt = []
  nega_cnt = []
  neu_cnt = []

  pos_dict, nega_dict, neu_dict = getDict(df)

  nouns = text_cleaning(s)

  cnt1 = 0
  cnt2 = 0
  cnt3 = 0

  for word in nouns:
    if word in pos_dict:
      cnt1 += 1
    if word in nega_dict:
      cnt2 += 1
    if word in neu_dict:
      cnt3 += 1

  pos_cnt.append(cnt1), nega_cnt.append(cnt2), neu_cnt.append(cnt3)

  transform_df = pd.DataFrame({
    'pos': pos_cnt,
    'nega': nega_cnt,
    'neu': neu_cnt
  })

  return transform_df


#KNeighborsClassifier 0.8507101843457238 {'fit__n_neighbors': 5, 'fit__weights': 'distance'}


def getModel(df):

  X = getX(df)

  e = LabelEncoder()
  y = e.fit_transform(df['극성'])

  X_train, X_test, y_train, y_test = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=13)

  # over sampling
  over_sampling_instance = SMOTE()
  o_X_train, o_y_train = over_sampling_instance.fit_resample(X_train, y_train)

  model = KNeighborsClassifier(n_neighbors=5, weights='distance')
  model.fit(o_X_train, o_y_train)

  return model
