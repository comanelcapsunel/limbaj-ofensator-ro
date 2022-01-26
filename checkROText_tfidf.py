import pandas as pd
from pandas import read_csv
from pandas import read_excel
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

### START ###

FN2 = "checkText.xlsx"

path="c:\WORK\=FEAA=\Master_DM\PDT\Project"
fn2 = path + "/" + FN2

# read data containing raw but also preprocessed text 
df_texts = pd.read_excel(fn2, sheet_name='texts');

print(f"Input file: {FN2}")
print()

# show descriptive plots

print(f"Label distribution")
print()
print(df_texts['Label'].value_counts())

plt.hist( df_texts.Label, bins = 5,)
plt.show()

print()
print(f"Offensive/Warning distribution on lenght: ")
print()
df_texts.head()
df_texts.isnull().sum()
df_texts['length']  = df_texts['Text'].str.len()
df_texts['Text'].value_counts()
warning = df_texts[df_texts['Label'] == 'warning']
warning.head()
offensive = df_texts[df_texts['Label'] == 'offensive']
offensive.head()
#notoffensive =df[df['Label'] == 'not offensive']

plt.hist(offensive['length'],label='offensive')
plt.hist(warning['length'], label='warning')

plt.legend()
plt.autoscale()
plt.show()

###
### Decision Tree on TF-IDF matrix
###

print()
print(f"Method: Classification with Decision Tree algorithm")
print()

vectorizer = TfidfVectorizer()
classifier = DecisionTreeClassifier()


# build TF-IDF matrix
X = vectorizer.fit_transform(df_texts['Text_Preprocesat'])
X=X.toarray()

# split data into 70% Training and 30% Test
X_train, X_test, Y_train, Y_test = train_test_split( X, df_texts['Label'], random_state=1234, test_size = 0.3, stratify= df_texts['Label'])

X_train.shape
Y_train.shape

X_test.shape
Y_test.shape

print(f"Data set split in 70% for Training and 30% for Testing")
print()

# train the model 
classifier.fit(X_train,Y_train)

# test the mode
predictions = classifier.predict(X_test)
predictions

# show results
acc = accuracy_score(Y_test, predictions)
print(f"Model accuracy: {acc}")
print()

cm = confusion_matrix(Y_test, predictions)
print(f"Confusion matrix:")
print(cm)
print()

print(classification_report(Y_test, predictions))

###
### Random Forest on TF-IDF matrix
###

print()
print(f"Method: Classification with Random Forest algorithm")
print()

from sklearn.ensemble import    RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100,n_jobs=-1)

clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)

# show results
acc2 = accuracy_score(Y_test, y_pred)
print(f"Model accuracy: {acc2}")
print()

cm2 = confusion_matrix(Y_test,y_pred)
print(f"Confusion matrix:")
print(cm2)
print()

#from mlxtend.plotting import plot_confusion_matrix
#plot_confusion_matrix(confusion_matrix(Y_test,y_pred))

print(classification_report(Y_test,y_pred))


###
### SVM on TF-IDF matrix
###

print()
print(f"Method: Classification with SVM algorithm")
print()

from sklearn.svm import SVC
clf2 = SVC(C = 1000, gamma='auto')

clf2.fit( X_train, Y_train)
y_pred = clf2.predict(X_test)

acc3 = accuracy_score(Y_test, y_pred)
print(f"Model accuracy: {acc3}")
print()

cm3 = confusion_matrix(Y_test,y_pred)
print(f"Confusion matrix:")
print(cm3)
print()

#from mlxtend.plotting import plot_confusion_matrix
#plot_confusion_matrix(confusion_matrix(Y_test,y_pred))

print(classification_report(Y_test,y_pred))


### END PROGRAM ###