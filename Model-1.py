import re
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.grid_search import GridSearchCV
#--------------------------------------------------------------
#Reading and Cleaning Data
ROOT_DIR = os.path.join( os.path.dirname( __file__ ))
input_file = os.path.join(ROOT_DIR,'Data.txt')
1
data = []
file = open(input_file,'r')
for line in file:
    text, label = line.split(',,,')                         #Separting text and Labels
    text = re.findall('[A-z]+', text)                       #Selecting Only Alphabets
    textTemp = [item.strip().lower() for item in text]      #Making Text to Lower Case
    text = ' '.join(textTemp)                               #Join the list to create a single sentence
    data.append([text, label.strip()])

df_data = pd.DataFrame(data)                                #Converting data to DataFrame
df_data.columns = ['Text','Label']                          #Giving Name to its Columns
#--------------------------------------------------------------
x = df_data['Text']
y = df_data['Label']
xTr,xTe,yTr,yTe = train_test_split(x,y,test_size=0.25,random_state=44)     #Select Random data for Train and Test

#---------------------------------------------------------------
#Creating feature and Model

pipeline = Pipeline([                                   #Putting all the operation in Pipeline. It will first find out
	('vect',  CountVectorizer(ngram_range=(1, 2))),     #bag of words Feature and put the result in vect variable
    #('tfidf', TfidfVectorizer()),                     #then it will calculates the tf-idf of features in vect
	('clf',  LogisticRegression()) ])
#---------------------------------------------------------------
#Training Data and Performing prediction on Test Data
pipeline.fit(xTr,yTr)
pred = pipeline.predict(xTe)

#Evaluating Results on sevelral measures
print 'Accuracy Score =', accuracy_score(yTe,pred)#,accuracy_score(yTr,pipeline.predict(xTr))
print '-'*60
print classification_report(yTe,pred)
#---------------------------------------------------------------



'''
Results--->>>

Accuracy Score = 0.962264150943
------------------------------------------------------------
             precision    recall  f1-score   support

affirmation       1.00      0.86      0.93        22
    unknown       0.93      0.91      0.92        58
       what       0.95      0.99      0.97       150
       when       0.96      0.86      0.91        28
        who       0.99      0.99      0.99       113

avg / total       0.96      0.96      0.96       371


'''