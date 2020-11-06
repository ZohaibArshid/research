# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:27:35 2020

@author: Zohaib Arshad Tanoli
"""
import numpy as np
import pandas as pd
from textaugment import EDA
from textaugment import Wordnet
import csv 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import classification_report

import nltk
#nltk.download('stopwords')

df= pd.read_csv('orignal dataset.csv', encoding= 'latin1')
df.head()
print(len(df))
a=[]
b=[]
df['Sentiment'].value_counts()
for i in range(370):
    sentences= df['Sentences']
#print(sentences)
    b.append(sentences[i])
    print(b)

s=[]
for i in range(370):
    sentiment=df['Sentiment']
    sent=sentiment[i]
    s.append(sent)
    print(sent)
    
a.append(b)
a.append(s)
print(len(a[0]))
print(len(a[1]))


def simple_split(a,y, length, split_mark=0.7):
    if split_mark > 0. and split_mark < 1.0:
        n= int(split_mark*length)
    else:
        n= int(split_mark)
    X_train = a[:n].copy()
    #print( X_train)
    X_test = a[n:].copy()
    #print( X_test)
    y_train = y[:n].copy()
    #print( y_train)
    y_test = y[n:].copy()
    #print( y_test)
    return X_train,X_test,y_train,y_test

vectorizer = CountVectorizer()
X_train,X_test,y_train,y_test= simple_split(a[0],a[1],len(a[0]))
print(len(X_train),len(X_test),len(y_train),len(y_test))

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names()
print(feature_names)

classifier= svm.SVC(kernel='linear', gamma='auto', C=2).fit(X_train, y_train)
y_predict = classifier.predict(X_test)
print(y_predict)
print(classification_report(y_test, y_predict))

















'''
ap=[]
for i in range(255):
    appname= df['Application']
    app=appname[i]
    ap.append(app)
    print(app)

#synomam Change
# wordnet is a english dictnory have lots of words with there synoniyams and antinonims
from textaugment import Wordnet
t = Wordnet()
aug=t.augment(str(sentences))
print(aug)


# Random Insertion
from textaugment import EDA
E= EDA()
insert=E.random_insertion(str(sentences))
print(insert)



# Random deletion
E= EDA()
delet=E.random_deletion(str(sentences))
print(delet)



# Random swap
E= EDA()
swap=E.random_swap(str(sentences))
print(swap)


a=[]
#for i in range(255):
E= EDA()
insert=E.random_insertion(str(b))
print(insert)
a.append(insert)
   #print(a)
print(len(insert))


data=[]
file=open('ahan.csv', 'w', newline='')
csv_writer = csv.writer(file, delimiter='|')
listdata=[a]
print(listdata)
data.append(listdata)
csv_writer.writecol(data)
'''








