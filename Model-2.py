import re
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#--------------------------------------------------------------
#Reading and Cleaning Data
ROOT_DIR = os.path.join( os.path.dirname( __file__ ))
input_file = os.path.join(ROOT_DIR,'Data.txt')

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

#--------------------------------------------------------------
#Creating Custom Features
#There is a pattern in the "what" class has "what" word in begining
#Similiraly "who" and "when"
#All affirmative sentences starts with an auxliary Verb
#And rest are Unknwn Class

#Just looking for that word as gien in class & its location and assgining 1 if it exists
#For Ex - "what are liver enzymes ? ,,, what"
#In the statement above word "what" is present at the start of sentence
#So 1 is assigned
#Similiarly for "Who" and "when"
#For Affirmative Sentences I am looking into the list of auxlilary verbs and assign 1 if exists
#rest all the cases are put into unknown


what =  [1 if 'what' in line and line.find('what') == 0 else 0 for line in df_data['Text']]
who = [1 if ('who' in line and line.find('who') == 0) else 0 for line in df_data['Text']]
when = [1 if (('when' in line) and (line.find('when') == 0)) or (line.find('what time') == 0) else 0
        for line in df_data['Text']]


aux_verb = ['is','am','are','was','were','will','shall','can','may','would','should','could','might',
            'do','did','does','has','have','had']
affirmative = [0]*len(x)
for index,line in enumerate(x):
    for av in aux_verb:
        if (line.find(av) == 0):
            affirmative[index] = 1
            break


unk = [0]*len(df_data['Text'])
for i in range(len(df_data['Text'])):
    if what[i] + who[i] + when[i] + affirmative[i] == 0:
        unk[i] = 1
#-----------------------------------------------------------------

n = pd.DataFrame({'what':what,'when':when,'who':who,'afi':affirmative,'unk':unk})
xTr,xTe,yTr,yTe = train_test_split(n,y,test_size=0.25, random_state=44)

#-----------------------------------------------------------------
#Training and Evaluation

lr = RandomForestClassifier()
lr.fit(xTr,yTr)
pred = lr.predict(xTe)
print 'Accuracy Score =', accuracy_score(yTe,pred)#,accuracy_score(yTr,lr.predict(xTr))
print '-'*60
print classification_report(yTe,pred)



'''
Results--->>>

Accuracy Score = 0.970350404313
------------------------------------------------------------
             precision    recall  f1-score   support

affirmation       1.00      1.00      1.00        22
    unknown       0.94      0.88      0.91        58
       what       0.99      0.97      0.98       150
       when       0.97      1.00      0.98        28
        who       0.96      1.00      0.98       113

avg / total       0.97      0.97      0.97       371


'''