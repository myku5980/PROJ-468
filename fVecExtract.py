from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy 
categories = ['artificial intelligence','computer science','python']
data_train = datasets.load_files('C:\\Users\\peter\\Documents\\dataset',description='This dataset contains the files extracted from the web to test documents classification with', 
    load_content=True, encoding='latin1' )

topicName = []
print(data_train.target)
for i in data_train.target:
    print(i)
    val = int(i)
    topicName.append(categories[val])
print(type(topicName))
print(topicName)



d = {'content':data_train.data, 'Topic':topicName,'Label':data_train.target}
df = pd.DataFrame(data=d)
print(df)

X_train, X_test, y_train, y_test = train_test_split(df['content'], df['Label'], random_state=1)
print("-------------------------------------------------------Training data: ")
print(X_train)
print(y_train)
print("-------------------------------------------------------")
print("-------------------------------------------------------Testing data: ")
print(X_test)
print(y_test)

print("-------------------------------------------------------")
print()

print("-------------------------------------------------------")




#-----------------------------------------------------------------------------------
countV = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
xTrainCounts = countV.fit_transform(X_train)
xTrainCounts.shape
print(xTrainCounts)
Xtestcv = countV.transform(X_test)
#-----------------------------------------------------------------------------------------------------------
tf_transformer = TfidfTransformer(use_idf=False).fit(xTrainCounts)
X_train_tf = tf_transformer.transform(xTrainCounts)
X_train_tf.shape

XtestcvTFIDF = tf_transformer.transform(Xtestcv)



print("--------------------------------Below we print feature vectors to export later")
word_freq_df = pd.DataFrame(xTrainCounts.toarray(), columns=countV.get_feature_names())
top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)

print(word_freq_df)
print(top_words_df)





print("----------------------------------------------------------------------------------")



















#---------------------------------------------------
print("Non TFIDF")
clf = MultinomialNB().fit(xTrainCounts, y_train)
pred = clf.predict(Xtestcv)

d2 = {'contents': X_test, 'test':y_test, 'Result':pred}
df = pd.DataFrame(data=d2)
print(df)

#------------------------------------------------------------------------------------

print("TFIDF")
clf_TFIDF = MultinomialNB().fit(X_train_tf, y_train)
pred = clf_TFIDF.predict(XtestcvTFIDF)

d2 = {'contents': X_test, 'test':y_test, 'Result':pred}
df = pd.DataFrame(data=d2)
print(df)
git add README.md

#2 1 2 1 0 0 0 1 0
#2 1 2 1 0 1 0 1 0


