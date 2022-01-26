import os
import sys
import xlwings as xw
import time

import pandas as pd
from pandas import read_csv
from pandas import read_excel
import numpy as np

import spacy
import ro_core_news_sm

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

### START ###

spacy.__version__
nlp = spacy.load('ro_core_news_sm')
nlp

FN1 = "badWords.xlsx"
FN2 = "checkText.xlsx"
#FN3 = "checkText.xlsx"

path="c:\WORK\=FEAA=\Master_DM\PDT\Project"
fn1 = path + "/" + FN1
fn2 = path + "/" + FN2
#fn3 = path + "/" + FN3
#a=read_csv('c:\\WORK\\=FEAA=\\Master_DM\\PDT\\Project\\badWords.xlsx')

# prepare excel-sheet for offensive / warning words
wb1 = xw.Book (fn1)
wsBadWords = wb1.sheets["words"]
#wsBadWords = read_csv('c:\\WORK\\=FEAA=\\Master_DM\\PDT\\Project\\badWords.xlsx')

# prepare excel-sheet for texts to check + reading params settings
wb2 = xw.Book (fn2)
wsCheck = wb2.sheets["texts"]
wsParam = wb2.sheets["params"]

#wsBadWords = read_csv('c:\\WORK\\=FEAA=\\Master_DM\\PDT\\Project\\badWords.xlsx')


# create lists for offensive words and warning words

listBadWords = wsBadWords.range("A1:B10000").value

listOffensive = [x[0] for x in listBadWords if x[0] != None and x[1] == "offensive"]
listWarning   = [x[0] for x in listBadWords if x[0] != None and x[1] == "warning"]

# create list for texts to check
listCheckTexts = wsCheck.range("A2:A10000").value
listCheckTexts = [x for x in listCheckTexts if x != None]
#type(listCheckTexts)

# read check parameters
countOffensiveWarning = int(wsParam["C1"].value)
countOffensive        = int(wsParam["C2"].value)
countWarningFrom      = int(wsParam["C4"].value)
countWarningTo        = int(wsParam["E4"].value)

def preprocessing(sentence):
    # a A
    sentence = sentence.lower()
    sentence = sentence.replace('.','')
    tokens = []
    tokens = [token.lemma_ for token in nlp(sentence) 
              if not (token.is_stop or token.like_num 
                      or token.is_punct 
                      or token.is_space
                      or len(token)==1)]
    tokens = ' '.join([element for element in tokens])
    return tokens

listCheckTexts_clean = list()

# Calculate and fill up "Count Offensive Words" and  "Count Warning Words" in excel file
for idxText, elemText in enumerate(listCheckTexts):
    # iterate to every row column A of the checkText.xlsx
    # check / count warnings and offensive words
    
    #idxText = 0
    #elemText = listCheckTexts[idxText]

    #print(f"idxText: {idxText}...")
    #print(f"Checking text in row {idxText + 2}...")
    elemText = preprocessing( elemText)
    listCheckTexts_clean.append(elemText)
    
    tmpCountOffensiveWords = tmpCountWarningWords = 0
    tmpFoundOffensive = []
    tmpFoundWarnings  = []
    
    for warningWord in listWarning:
      if warningWord in elemText:
        tmpCountWarningWords += 1
        tmpFoundWarnings.append(warningWord)
        
    for offensiveWord in listOffensive:
      if offensiveWord in elemText:
        tmpCountOffensiveWords += 1
        tmpFoundOffensive.append(offensiveWord)
    
    #print(f"Found {tmpCountWarningWords} warning words: {tmpFoundWarnings}...")
    #print(f"Found {tmpCountOffensiveWords} offensive words: {tmpFoundOffensive}...")
    
    # write the results for the row in excel
    #wsCheck[f"E{idxText + 2}"].value = tmpResult
    wsCheck[f"C{idxText + 2}"].value = tmpCountOffensiveWords
    wsCheck[f"D{idxText + 2}"].value = tmpCountWarningWords
    wsCheck[f"F{idxText + 2}"].value = elemText
    #print(f"Text in row {idxText} contains {tmpCountOffensiveWords} offensive words and {tmpCountWarningWords} warning words")

wb2.save()


def calculateAccuracy(fileName):
    data = pd.read_excel(fileName, sheet_name='texts');
                   #dtype={"TAIL_NUMBER": "string", "ORIGIN_AIRPORT": "string", "DESTINATION_AIRPORT": "string"}) 
    tab = pd.crosstab(data.Label, data.Classification, margins=False)
    return np.diag(tab).sum() / tab.to_numpy().sum()


#  countOffensiveWarning = int(wsParam["C1"].value)
#  countOffensive        = int(wsParam["C2"].value)
#  countWarningFrom      = int(wsParam["C4"].value)
#  countWarningTo        = int(wsParam["E4"].value)

df_texts = pd.read_excel(fn2, sheet_name='texts');

bestParams = pd.DataFrame([{'accuracy':0, 'countOffensive':0, 'countOffensiveWarning':0, 'countWarningFrom':0}])
bestAccuracy = 0
#bestParams[bestParams['accuracy']==bestAccuracy].countOffensive

for i_countOffensive in range(1,countOffensive+1):
    for i_countOffensiveWarning in range(2,countOffensiveWarning+1):
        for i_countWarningFrom in range(1,i_countOffensiveWarning):
            #print(f"countOffensiv: {i_countOffensive} countOffensiveWarning:{i_countOffensiveWarning} countWarningFrom: {i_countWarningFrom}")
            #i_countOffensive = 1
            #i_countOffensiveWarning = 4
            #i_countWarningFrom = 3
            
            df_texts.Classification = 'not offensive'
            df_texts.loc[ (df_texts['Count Warning Words'] >= i_countWarningFrom) & (df_texts['Count Warning Words'] < i_countOffensiveWarning), 'Classification'] = 'warning'
            df_texts.loc[ df_texts['Count Warning Words'] >= i_countOffensiveWarning, 'Classification'] = 'offensive'
            df_texts.loc[ df_texts['Count Offensive Words'] >= i_countOffensive, 'Classification'] = 'offensive'
            df_texts.loc[ (df_texts['Classification'] == 'classification') & (df_texts['Count Offensive Words'] > 1), 'Classification'] = 'offensive'
            
            #df_texts[['Label','Count Warning Words','Count Offensive Words','Classification']]
            
            tab = pd.crosstab(df_texts.Label, df_texts.Classification, margins=False)
            acc = np.diag(tab).sum() / tab.to_numpy().sum()
            #acc
            #bestParams
            if acc > bestAccuracy:
                bestParams = pd.DataFrame([{'accuracy':acc, 'countOffensive':i_countOffensive, 'countOffensiveWarning':i_countOffensiveWarning, 'countWarningFrom':i_countWarningFrom}])
                bestAccuracy = acc
                #best_df_texts = df_texts
            #bestParams

#print(f"Found best accuray of {bestAccuracy} for:")       
best_countOffensive = bestParams.loc[0,'countOffensive']   
#print(f"countOffensive: {best_countOffensive}")    
best_countOffensiveWarning = bestParams.loc[0,'countOffensiveWarning']   
#print(f"countOffensiveWarning: {best_countOffensiveWarning}")   
best_countWarningFrom = bestParams.loc[0,'countWarningFrom']   
#print(f"countWarningFrom: {best_countWarningFrom}")   

print(f"Input file: {FN2}")
print()
print(f"Method: Classification based on Warning/Offensive counts")
print()
print(f"countWarningFrom: {best_countWarningFrom}")
print(f"countOffensiveWarning: {best_countOffensiveWarning}")
print(f"countOffensive: {best_countOffensive}")
print()
print(f"Classication rules:")
print(f"Texts containing at least {best_countWarningFrom} but less than {best_countOffensiveWarning} Warning words and no Offensive words are classified as \"Warning\"")
print(f"Texts containing at least {best_countWarningFrom} Warning words and at least one Offensive words are classified as \"Offensive\"")
print(f"Texts with more than {best_countOffensiveWarning} Warning words or more than {best_countOffensive} Offensive words are classified as \"Offensive\"")
print(f"Texts no falling under any of the above rules are classified as \"Not Offensive\"")


# use the found parameters to write the predicted classification into excel file 
for idxText, elemText in enumerate(listCheckTexts):
    #idxText = 0
    #elemText = listCheckTexts[idxText]
  
    # get the number of offensive and warning words for each row 
    countOffensiveWords     = int(wsCheck[f"C{idxText + 2}"].value)
    countWarningWords       = int(wsCheck[f"D{idxText + 2}"].value)

    tmpResWarning   = countWarningWords >= best_countWarningFrom and countWarningWords < best_countOffensiveWarning
    tmpResOffensive = countOffensiveWords >= best_countOffensive or countWarningWords >= best_countOffensiveWarning or (tmpResWarning and countOffensiveWords > 0)

    if tmpResOffensive:
        tmpResult = "offensive"
    else:
        if tmpResWarning:
            tmpResult = "warning"
        else:
            tmpResult = "not offensive"
            
    # write the results for the row in excel
    wsCheck[f"E{idxText + 2}"].value = tmpResult
    #print(f"Text in row {idxText+2} is {tmpResult}")
    #print(f"Text in row {idxText} contains {tmpCountOffensiveWords} offensive words and {tmpCountWarningWords} warning words")

wb2.save()
wb2.close()
wb1.close()

# read back excel file and compute confusion matrix based on Label (observed) and Classification (predicted)
df_texts = pd.read_excel(fn2, sheet_name='texts');

acc = accuracy_score(df_texts['Label'], df_texts['Classification'])
print(f"Model accuracy: {acc}")
print()

cm = confusion_matrix(df_texts['Label'], df_texts['Classification'])
print(f"Confusion matrix:")
print(cm)
print()

print(classification_report(df_texts['Label'], df_texts['Classification']))

### END PROGRAM ###



