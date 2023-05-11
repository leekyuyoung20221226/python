# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:14:07 2023
FileName : .py
@author: user
"""

    
    
import music21
little_star="tinynotation: 4/4 c4 c4 g4 g4 a4 a4 g2 f4 f4 e4 e4 d4 d4 c2 g4 g4 f4 f4 e4 e4 d2 g4 g4 f4 f4 e4 e4 d2 c4 c4 g4 g4 a4 a4 g2 f4 f4 e4 e4 d4 d4 c2"
#music21.converter.parse(little_star).show('mid')

# ABC 표기법은 2채널(계이름, 박자), 분류 문제
# 특징벡터 [[1,4],[1,4],[5,4] ...] 
# 레이블 [5,4] --> 원핫벡터
import numpy as np
# 딕셔너리 구조를 이용해서 계이름과 <--> 박자
note2num = {'c':1,'d':2,'e':3,'f':4,'g':5,'a':6,'b':7}
num2note =  {value:key for key,value in note2num.items() }
print(note2num); print(num2note)

# ABC 표기를 시계열 데이터로 변환
def abc2timeserial(s):
    notes =  s.split(' ')[2:]
    seq=[]
    for i in notes:
        seq.append( [note2num[i[0]], int(i[1]) ])
    return seq

# 시계열 데이터를 ABC 표기로 변환
def timeserial2abc(t):
    s = "tinynotation: 4/4"
    for i in t:
        s = s + ' ' + num2note[ i[0]  ] + str(i[1])
    return s

# 원핫 코드로 변경

# 시계열 데이터를 훈련 집합으로 자름
def seq2dataset(seq, window,horizon):
    x,y = [],[]
    for i in range(len(seq) - (window+horizon)+1):
        x_ = seq[i:(i+window)]
        y_ = seq[i+window+horizon-1]
        x.append(x_); y.append(y_)
    return np.array(x), np.array(y)
w = 8
h = 1 
seq = abc2timeserial(little_star)
x,y = seq2dataset(seq, w, h)
print(x.shape, y.shape)

# 훈련 집합 구축 - 예측목적이 아니라 생성목적 전체를 훈련 집합으로 사용
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#import tensorflow as  tf
#from sklearn.model_selection import train_test_split

split = int(len(x)*1.0)
x_train = x[:split]; y_train = y[:split]
print(x_train.shape, y_train.shape)

# 원핫인코딩
onehot = [[i,2] for i in range(1,8)] + [[i,4] for i in range(1,8)] + [[i,8] for i in range(1,8)]
def to_onehot(data):
    t = []
    for i in range(len(data)):
        a = np.zeros((len(onehot)))
        a[onehot.index(list(data[i]))] = 1.0
        t.append(a)
    return np.array(t)
y_train = to_onehot(y_train)
print(x_train[0].shape, y_train.shape[1])

# LSTM모델 설계 및 학습
model = Sequential()
model.add(LSTM(128,activation='relu', input_shape=x_train[0].shape))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=200,batch_size=1, verbose=2)

# 학습된 모델로 편곡을 하는 함수를  생성
# first_measure : 첫 소절
# duration : 생성될 곡의 길이 
def arranging_music(model, first_measure, duration):
    music = first_measure # 첫소절을 지정
    for i in range(duration): # 반복
        # 직전 윈도우를 보고 다음 음표를 예측
        p = model.predict(np.float32(np.expand_dims(music[-w:], axis=0)))        
        # 가장 큰 확률을 가진 부로로 분류에 추가하고
        music = np.append(music, [onehot[np.argmax(p)]],axis=0)        
    return timeserial2abc(music) # ABC표기로 변환

new_song = arranging_music(model, x_train[0],50)
print(new_song);print(little_star)
music21.converter.parse(new_song).show('mid')

# 변형된 첫 소절로 편곡

new_song = arranging_music(model, 
 [[1,4],[2,4],[3,4],[5,8],[4,8],[3,8],[2,2],[2,4],[1,4],[2,4],[3,4],[5,8],[4,8],[3,8],[2,2],[2,4]]
,50)
music21.converter.parse(new_song).show('mid')


# 악보
stream = music21.converter.parse(little_star)
music21.environment.set('musicxmlPath', 'C:/Program Files/MuseScore 3/bin/MuseScore3.exe')

stream.write('musicxml.png', fp='C:/Program Files/MuseScore 3/bin/MuseScore3.exe')


# 여러곡을 학습해서 결합한 새로운 곡을 생성
spring_picnic="tinynotation: 4/8 g8 e8 g8 e8 g8 a8 g4 e8 g8 e8 c8 d8 e8 c4 g8 e8 g8 e8 g8 a8 g4 b8 a8 g8 e8 d8 e8 c4"
butterfly="tinynotation: 2/4 g8 e8 e4 f8 d8 d4 c8 d8 e8 f8 g8 g8 g4 g8 e8 e8 e8 f8 d8 d4 c8 e8 g8 g8 e8 e8 e4 d8 d8 d8 d8 d8 e8 f4 e8 e8 e8 e8 e8 f8 g4 g8 e8 e4 f8 d8 d4 c8 e8 g8 g8 e8 e8 e4"

# 3개의 곡을 시계열로 변환하고 결합
seq1 = abc2timeserial(little_star)
seq2 = abc2timeserial(butterfly)
seq3 = abc2timeserial(spring_picnic)
seq = seq1+seq2+seq3

x,y = seq2dataset(seq, w, h)












