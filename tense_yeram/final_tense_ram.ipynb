{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "과거\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# 형태소만 뽑아 저장\n",
    "stopwords = ['은','는','가','하','아','것','들','의','되','수','보','주','등','한']\n",
    "\n",
    "def tense_tokenizer(sentence, pos=[\"J\",\"E\"], stopword=stopwords): \n",
    "\n",
    "    from konlpy.tag import Hannanum\n",
    "    hannanum = Hannanum()\n",
    "\n",
    "    sentence = [word for word, tag in hannanum.pos(sentence) if len(word) > 0 and tag in pos and word not in stopword]\n",
    "\n",
    "    return sentence\n",
    "\n",
    "\n",
    "# tfidf, model\n",
    "def tense_ml(sentence):\n",
    "\n",
    "    import joblib\n",
    "\n",
    "    loaded_vectorizer = joblib.load('./tense_vectorizer_customized.pkl') \n",
    "    loaded_model = joblib.load('./tense_model_customized.pkl') \n",
    "\n",
    "    #df_token_list = [str(x).lower() for x in df]\n",
    "    df_token = tense_tokenizer(sentence)       \n",
    "    df_tfidf = loaded_vectorizer.transform(df_token)\n",
    "\n",
    "    pred = loaded_model.predict(df_tfidf)\n",
    "    pred = pred[-1]\n",
    "\n",
    "    return pred\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    print(tense_ml('김금희 소설가는 ＂계약서 조정이 그리 어려운가 작가를 격려한다면서 그런 문구 하나 고치기가 어려운가 작가의 노고와 권리를 존중해줄 수 있는 것 아닌가＂라고 꼬집었다.'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "ds_study",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.15"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "189257d3ead7ed4cabfbed122a0ee4e0c566f64830e9ec3d7b4f7f74d108b2bc"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
