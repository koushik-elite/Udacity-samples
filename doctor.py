import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

# training_set = pd.read_excel("Final_Train.xlsx", keep_default_na=False)
# test_set = pd.read_excel("Final_Test.xlsx", keep_default_na=False)

train = pd.read_csv('X_train.csv', keep_default_na=False)
test = pd.read_csv('X_test.csv', keep_default_na=False)
y = pd.read_csv('y_train.csv', keep_default_na=False)

# change all the column of orients to StandardScaler
for col in train.columns:
    if 'orient' in col:
        scaler = StandardScaler()
        train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1))
        test[col] = scaler.fit_transform(test[col].values.reshape(-1, 1))

# group by series id
# transpose the rows to column before sending to the Deep learning
train = np.array([x.values[:,3:].T for group,x in train.groupby('series_id')], dtype='float32')
test = np.array([x.values[:,3:].T for group,x in test.groupby('series_id')], dtype='float32')

le = LabelEncoder()
y = le.fit_transform(y['surface'])

# print(y)

oof = np.zeros((len(train), 9))
prediction = np.zeros((len(test), 9))
scores = []

print(oof)
print(prediction)

for fold_n, (train_index, valid_index) in enumerate(folds.split(train, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = train[train_index], train[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

exit()

# training_set.to_csv("Final_Train.csv", index= False)
# training_set = pd.read_csv("Final_Train.csv")
# print(training_set)

# for i,f in enumerate(training_set.columns):
    # print(f)
    # training_set[f] = le.fit_transform(training_set[f])
    # if(X_test[f]):
        # X_test[f] = le.transform(X_test[f])

le = preprocessing.LabelEncoder()
# training_set = training_set.apply(le.fit_transform)

le_test = preprocessing.LabelEncoder()
# test_set = test_set.apply(le_test.fit_transform)

X_train = training_set.iloc[:,0:6].values
Y_train = training_set.iloc[:,-1].values
X_test = test_set.iloc[:,0:6].values

lables = list()

for x in X_train[:, 0]:
    for word in x.split(','):
        lables.append(word.strip().lower())

lables = Counter(lables)
print(lables)

# lables = set([ for x in X_train[:, 0]])

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# imp = SimpleImputer(missing_values=np.nan, strategy="mean")
# X_train = imp.fit_transform(X_train)
# Y_train = imp.fit_transform(Y_train)

scaler = StandardScaler()

# print(X_train[5961, 6])
# X_train = le.fit_transform(X_train).astype(str)


# Data Preprocessing
"""Follow the instructions at :
https://www.analyticsindiamag.com/data-pre-processing-in-python/ to perform Data-Preprocessing or Cleaning"""

# Using Decision Tree Regressor to Predict the Doctor Fees
#importing the library for Decision Tree Regressor

print(le.classes_)
# print(Y_train)

#Initializing the Deision Tree Regressor
dtr = DecisionTreeRegressor()

# Fitting the Decision Tree Regressor with training Data
dtr.fit(X_train,Y_train)
Y_pred_dtr = dtr.predict(X_test)
Y_pred_dtr = le.inverse_transform(Y_pred_dtr)

print(Y_pred_dtr)

# Predicting the values(Fees) for Test Data

# Y_pred_dtr = scaler.inverse_transform(dtr.predict(X_test))
"""NOTE:scaler.inverse_transform is used to inverse the scaling"""

# Saving the Predicted values in to an Excel file
pd.DataFrame(Y_pred_dtr, columns = ['Fees']).to_excel("predictions_tree.xlsx", index = False)