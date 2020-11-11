import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization

df = pd.read_csv('c:/users/kevin/desktop/bike_buyers_clean.csv')

#print(df.head())

#print(df.shape)

#print(df.columns)

#print(df.info())

corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
#plt.show()

#Analyzing Numerical Variables

numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))
print('The numerical variables are :', numerical)

#print(df[numerical].head())

# check missing values in numerical variables
#print(df[numerical].isnull().sum())

categorical = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))
print('The categorical variables are :', categorical)

#print(df[categorical].head())
#print(df[categorical].isnull().sum())

# view frequency counts of values in categorical variables
for var in categorical:
    print(df[var].value_counts())
    print(df[var].value_counts()/np.float(len(df)))
    print()

# check for cardinality in categorical variables
for var in categorical:
    print(var, ' contains ', len(df[var].unique()), ' labels')

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

df['Marital Status'] = label_encoder.fit_transform(df['Marital Status'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Education'] = label_encoder.fit_transform(df['Education'])
df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
df['Home Owner'] = label_encoder.fit_transform(df['Home Owner'])
df['Commute Distance'] = label_encoder.fit_transform(df['Commute Distance'])
df['Region'] = label_encoder.fit_transform(df['Region'])
df['Purchased Bike'] = label_encoder.fit_transform(df['Purchased Bike'])
df.head()

#print(df['Age'].describe())

df['Age'] = pd.cut(x = df['Age'], bins = [0,30,40,50,60,100,150], labels = [0, 1, 2, 3, 4, 5])
df['Age'] = df['Age'].astype('int64')
df['Age'].isnull().sum()

#print(df['Income'].describe())

df['Income'] = pd.cut(x = df['Income'], bins = [0, 30000, 50000, 75000, 100000, 150000, 200000], labels = [1, 2, 3, 4, 5, 6])
df['Income'] = df['Income'].astype('int64')
df['Income'].isnull().sum()

print(df.dtypes)

#Train-Test Split

X = df.drop(['Purchased Bike'], axis=1)
y = df['Purchased Bike']

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 999)
X_train.shape, X_test.shape

#print(X_train.head())
#print(X_train.shape)

#print(X_test.head())
#print(X_test.shape)

#Gausian Naive Bayes
# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB

# instantiate the model
gnb = GaussianNB()

# fit the model
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

y_pred[:10]
len(y_pred)

from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = gnb.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n', cm)

_matrix = pd.DataFrame(data=cm, columns=['Predict Positive', 'Predict Negative'],
                                 index=['Actual Positive', 'Actual Negative'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
