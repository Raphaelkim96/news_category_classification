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

df = pd.read_csv('./crawling_data/naver_headline_news_20241219_combined_data.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df.info()

print(df.category.value_counts())

#전처리

X = df['titles']
Y = df['category']
okt = Okt()

#라벨링
encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
print(labeled_y[:3] )

label = encoder.classes_
print(label)

#encoder 저장
with open('./models/encoder.pickle', 'wb') as f: #쓰기 바이너리
    pickle.dump(encoder, f) #f파일에 저장

# label 원핫 인코딩
onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

# x 문장데이터 처리, 불용어 제거

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

token = Tokenizer()
token.fit_on_texts(X)
print(X[:5])
tokened_X = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1
print(wordsize)
print(tokened_X[:5])

# 최대값 찾기 알고리즘
max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])
print(max)


# 문장들 길이 맞추기, 짧은거 0 추가

X_pad = pad_sequences(tokened_X, max)
print(X_pad)
print(len(X_pad[0]))

# train,test split
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

np.save('./crawling_data/news_data_X_train_max_{}_wordsize_{}'.format(max,wordsize), X_train)
np.save('./crawling_data/news_data_Y_train_max_{}_wordsize_{}'.format(max,wordsize), Y_train)
np.save('./crawling_data/news_data_X_test_max_{}_wordsize_{}'.format(max,wordsize), X_test)
np.save('./crawling_data/news_data_Y_test_max_{}_wordsize_{}'.format(max,wordsize), Y_test)
