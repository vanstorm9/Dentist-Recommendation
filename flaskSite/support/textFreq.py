import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.pipeline import FeatureUnion
from nltk.corpus import stopwords 
import os
import string

import heapq 

doctorData = './data/data_hack.csv'


def removeStopWords(wordList):
    return [word for word in wordList if word not in stopwords.words('english')]


def calcCosSim(a,b):
    # Calculate cosine simularity
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def removeStem(sentence):
    ps = PorterStemmer()
    words = word_tokenize(sentence)
    tmpStr = ''
    for w in words:
        tmpStr += ps.stem(w) + ' '
    return tmpStr



def recommendDoctor(searchQuery,topNNum):
    
    df = pd.read_csv(doctorData)

    doctorMainAr = df[['spec','treatment','insurance','dental clinic']].values

    customerVec = removeStopWords([removeStem(searchQuery.lower())])

    # Preparing customer vector
    vec = CountVectorizer()
    customerFreq = vec.fit_transform(customerVec)
    customerDf = pd.DataFrame(customerFreq.toarray(),columns=vec.get_feature_names())

    resScore = []
    #resPara = []

    for i in range(0,len(doctorMainAr)):
        doctorStr = doctorMainAr[i][0] + ' ' + doctorMainAr[i][1] + ' ' + doctorMainAr[i][2]
        doctorVec = removeStopWords([removeStem(doctorStr.lower())])
        
        # Prepare to doctor vector to combine vector
        doctorFreq = vec.fit_transform(doctorVec)
        doctorDf = pd.DataFrame(doctorFreq.toarray(),columns=vec.get_feature_names())
        
        # Combine vectors
        combinedDf = pd.concat([customerDf, doctorDf],sort=False).fillna(value=0.0)
        customerVec = combinedDf.iloc[0].values
        doctorVec = combinedDf.iloc[1].values
        
        # Preform cosine simularity
        simRes = calcCosSim(customerVec,doctorVec)
        
        # Appending results
        resScore.append(simRes)
        #resPara.append(doctorStr)


    rankAr = np.asarray(resScore).argsort()[::-1][:]
    heapLi = []

    for ind in rankAr:    
        heapq.heappush(heapLi,[-resScore[ind],ind])
       

    resAr  = [] 
    i = 0
    while len(heapLi) > 0 and i < topNNum:
        score,ind = heapq.heappop(heapLi)
        score = score*-1
        
        if score < 0.01:
          # Results from now on are not relevent
          break
        
        resAr.append([doctorMainAr[ind]][0])


    return resAr





