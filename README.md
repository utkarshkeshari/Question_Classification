# Question_Classification

### Package Dependecies

* re,os,pandas,scikit-learn

### Data Cleaning

* Removed all the digits and symbols

* Converted the alphabets to lower case

### Two models are developed

##### Model-1.py
In this File, unigram and bigram features are being created
and this features are directly being provided to LogisticRegression for training
After Prediction Accuracy Score is 96.22% and f1-score is 0.96

##### Model-2.py
In this file, Few custom features* are being created
and then peformed RandomForest for clasification
Accuracy Score is 97.03% and f1-score is 0.97

**Custom Feature**
There is a pattern seen in the data. -- All the "what","who" and "when" class has the same(as label) word in the data provided and also these words are at the begining of the sentence. -- All the "Affirmation" class has an auxilary verb at the begining of each sentence. -- And the classes doesn't belong to these classes are unknown class

Hence, One Hot Encoder feature is created using the pattern listed above
