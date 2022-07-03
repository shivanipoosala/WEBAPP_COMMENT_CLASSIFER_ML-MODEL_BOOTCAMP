import pandas as pd

import nltk #Natural Language Tool Kit
nltk.download(['punkt', 'wordnet'])
nltk.download('omw-1.4')

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
def tokenize(text):

    tokens = word_tokenize(text)
    #print("By doing Tokenization:")
    #print(tokens)

    lemmatizer = WordNetLemmatizer() #converts the words(tokens) into their base form wrto the context by mapping with WordNet Cloud

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
   # print("After Performing Lemmatization:")
    return clean_tokens


path = "dataset.json"


def load_data(path):
    # reading the datset
    df = pd.read_json(path, lines=True)
    # print(df)
    # cleaning the dataset
    df["label"] = df.annotation.apply(lambda x: x.get('label'))
    df["label"] = df.label.apply(lambda x: x[0])

    # loading the data
    X = df.content.values  # Independent var
    y = df.label.values  # dependent var

    return X, y

url = 'dataset.json'
X,y=load_data(url)
#split dataset

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0,shuffle=True)
#print(X_train)

vect = CountVectorizer(tokenizer=tokenize) #generates frequency table for tokens in all the documents (comments) - lemmatization is also applied
tfidf = TfidfTransformer()
clf = DecisionTreeClassifier()

#train our model - train classifier
X_train_count = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_count)
clf.fit(X_train_tfidf,y_train) #fitting the algorithm with vectors (converted by applying Countvectorization and tfidf)

#test our model
X_test_count = vect.transform(X_test)
X_test_tfidf = tfidf.transform(X_test_count)
y_pred = clf.predict(X_test_tfidf)

#test accuracy
print("Test Accuracy:")
print(clf.score(X_test_tfidf,y_test))

X1=vect.transform(X)
X1_tfidf = tfidf.transform(X1)
print(clf.score(X1_tfidf,y))

#serialization
pickle.dump(clf,open('model.pkl','wb'))
print("Succesfully Created")