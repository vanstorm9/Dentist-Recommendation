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


complaintSubPath = 'complaintMain/'


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



def recommendDoctor(topNRankNum,splitNum,rootPath):
    contractPath = rootPath + 'termService/' + '0.txt'
    print(rootPath + complaintSubPath)
    contractTxtOrg = open(contractPath, encoding="utf8", errors='ignore').read()
    contractTxt = contractTxtOrg
    contractVec = removeStopWords([removeStem(contractTxt)])




    vec = CountVectorizer()
    contractFreq = vec.fit_transform(contractVec)



    compList = []

    # Used for removing stem from words
    for cFilePath in os.listdir(rootPath + complaintSubPath):
        cFilePath = rootPath + complaintSubPath + cFilePath
        cFileTxt = open(cFilePath, encoding="utf8", errors='ignore').read()

        cFileTxt = removeStem(cFileTxt)
        
        compList.append(cFileTxt + ' ')
        

    compList = [''.join(str(v) for v in compList)]  # We are combining all complaints into one
    complantCnt = CountVectorizer()
    a = complantCnt.fit_transform(compList)

    resPara = []
    resScore = []

    
    #for paragraph in contractTxt.split('\n'):
    for paragraph in [contractTxt[i:i+splitNum] for i in range(0, len(contractTxt), splitNum)]:
        paragraph = paragraph.translate(string.punctuation)


        try:
            vec = CountVectorizer()
            contractFreq = vec.fit_transform([paragraph])
        except:
            continue
        complaintDf = pd.DataFrame(a.toarray(),columns=complantCnt.get_feature_names())
        contractDf = pd.DataFrame(contractFreq.toarray(),columns=vec.get_feature_names())

        combinedDf = pd.concat([complaintDf, contractDf],sort=False).fillna(value=0.0)
        complainVec = combinedDf.iloc[0].values
        contractVec = combinedDf.iloc[1].values

        # Now we calculate cosine simularity and append these values
        simRes = calcCosSim(complainVec,contractVec)
        resPara.append(paragraph)
        resScore.append(simRes)


    # Now we calculate the top rank

    rankAr = np.asarray(resPara).argsort()[::-1][:topNRankNum]

    heapLi = []


    for ind in rankAr:    
        heapq.heappush(heapLi,[-resScore[ind],ind])
    
    resText = []
    resRank = []

    while len(heapLi) > 0:
        score,ind = heapq.heappop(heapLi)
        score = score*-1
        
        resText.append(resPara[ind])
        resRank.append(score)


    return resText,resRank, contractTxtOrg.split('\n')





