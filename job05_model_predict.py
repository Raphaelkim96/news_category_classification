import pickle # 타입변환 없이 binary 코드로 저장

import pandas as pd #dataFrame 문자열을 맞는 데이터타입으로 변환
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
#한국어 토큰나이저
# jpype1 설치
from konlpy.tag import Okt,Kkma  #open korean tokenizer,java 기반
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.api.models import load_model

df = pd.read_csv('./crawling_data/naver_headline_news_0_1_20241223.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df.info()

print(df.category.value_counts())

#전처리

X = df['titles']
Y = df['category']


#라벨링

#encoder 불러오기
with open('./models/encoder.pickle', 'rb') as f: #읽기 바이너리
    encoder = pickle.load(f)

label = encoder.classes_
print(label)

labeled_y = encoder.transform(Y) # label이 이미 정해져있으면  fit_transform 말고 transform
# label 원핫 인코딩
onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

# x 문장데이터 처리, 불용어 제거
okt = Okt()
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
print(X)

# stopwords 불용어 다운로드
stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)
print(stopwords)

for sentence in range(len(X)):
    words = []
    for word in range(len(X[sentence])):
        if len(X[sentence][word]) > 1:
            if X[sentence][word] not in list(stopwords['stopword']):
                words.append(X[sentence][word])
    X[sentence] = ' '.join(words)

print(X[:5])

# 토큰나이징

# 저번에 했던 형태소 변환으로 써야 된다. 오늘 새로운 형태소가 있으면 0으로 처리해야됨
with open('./models/news_token_max_16.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)


# token = Tokenizer()
# token.fit_on_texts(X) # 형태소 라벨링
# print(X[:5])
# tokened_X = token.texts_to_sequences(X)
# wordsize = len(token.word_index) + 1
# print(wordsize)
# print(tokened_X[:5])

for i in range(len(tokened_X)):
    if len(tokened_X[i])>16: # 16보다 크면
        tokened_X[i] = tokened_X[i][:16] # 0~15만 저장

print(tokened_X[:5])
# 최대값 찾기 알고리즘
# max = 0
# for i in range(len(tokened_X)):
#     if max < len(tokened_X[i]):
#         max = len(tokened_X[i])
#
# print(max)



# 문장들 길이 맞추기, 짧은거 0 추가

X_pad = pad_sequences(tokened_X, 16)
print(X_pad[:5])
print(len(X_pad[0]))

# 제일 높은 확률 모델 불러오기
model = load_model('./models/news_category_classification_model_0.6774193644523621.h5')

# 오늘 데이터 예측하기
preds = model.predict(X_pad)

# 예측한 라벨 데이터프레임에 추가하기, 두번째 큰값 = 최댓갑 지우기
predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)]=0
    second = label[np.argmax(pred)]
    predicts.append([most, second])
df['predict']= predicts

print(df.head(30))

# 예측결과 수치 보기
score = model.evaluate(X_pad, onehot_Y)
print(score[1])

# 첫번째 큰 예측 라벨이 맞으면 1인 데이터 추가
df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i,'predict'] :
        df.loc[i,'OX'] = 1
#print(df.head(30))
print(df.OX.mean())


