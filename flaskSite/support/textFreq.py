import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.pipeline import FeatureUnion
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import os
import string

import heapq 

doctorData = './data/data_hack.csv'


# Parse arrays


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
  
  
def parseStrList(inputStr,appendChar=' '):
  resAr = inputStr.split('|')
  resStr = appendChar.join(resAr)

  return resStr, resAr

def vocabExtender(vocList):
  resStr = ""
  if 'unitedhealthcare' in vocList:
    resStr += "united healthcare "
   
  return resStr

def vocabExtenderList(vocList):
  resStr = ""
  for tmpEle in vocList:
    if tmpEle == 'crowns':
      resStr += "crown"
    if tmpEle == 'dental implants':
      resStr += "implant"
      resStr += "dental implant"
   
  return resStr

def sentiment_analyzer_scores(sentence):
  analyser = SentimentIntensityAnalyzer()
  score = analyser.polarity_scores(sentence)
  return score


def recommendDoctor(searchQuery,topNNum):
    
    df = pd.read_csv(doctorData)

    doctorMainAr = df[['spec','treatment','insurance','dental clinic','language','phone','review']].values

    customerVec = removeStopWords([removeStem(searchQuery.lower())])

    # Preparing customer vector
    vec = CountVectorizer()
    customerFreq = vec.fit_transform(customerVec)
    customerDf = pd.DataFrame(customerFreq.toarray(),columns=vec.get_feature_names())

    resScore = []


      # Preparing customer vector
    vec = CountVectorizer()
    customerFreq = vec.fit_transform(customerVec)
    customerDf = pd.DataFrame(customerFreq.toarray(),columns=vec.get_feature_names())

    resScore = []

    for i in range(0,len(doctorMainAr)):
        treatStr, treatAr = parseStrList(doctorMainAr[i][1])
        insureStr, insureAr = parseStrList(doctorMainAr[i][2])
        langStr, langAr = parseStrList(doctorMainAr[i][4])

        doctorStr = doctorMainAr[i][0] + ' ' + doctorMainAr[i][1] + ' ' + doctorMainAr[i][2] + ' ' + insureStr + ' ' + doctorMainAr[i][4] + ' ' + langStr + ' '

        doctorStr += vocabExtender(insureAr)
        doctorStr += vocabExtenderList(treatAr)

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

        if isinstance(doctorMainAr[i][6], str):
          emotionRating = sentiment_analyzer_scores(doctorMainAr[i][6])
          simRes += (emotionRating['pos']*0.3 - emotionRating['neg']) 


        # Appending results
        resScore.append(simRes)



    rankAr = np.asarray(resScore).argsort()[::-1][:]
    heapLi = []

    for ind in rankAr:    
        heapq.heappush(heapLi,[-resScore[ind],ind])

    resAr = [] 

    i = 0
    while len(heapLi) > 0 and i < topNNum:
        score,ind = heapq.heappop(heapLi)
        score = score*-1

        if score < 0.01:
          # Results from now on are not relevent
          break

        # Break down list of insurance
        symptStr, symptAr = parseStrList(doctorMainAr[ind][1],',')
        insureStr, insureAr = parseStrList(doctorMainAr[ind][2],',')
        languageStr, languageAr = parseStrList(doctorMainAr[ind][4],',')

        doctorMainAr[ind][1] = symptStr
        doctorMainAr[ind][2] = insureStr
        doctorMainAr[ind][4] = languageStr


        if not isinstance(doctorMainAr[ind][6], str):
           doctorMainAr[ind][6] = ''

        resAr.append(doctorMainAr[ind])

        i += 1
       

    return resAr





