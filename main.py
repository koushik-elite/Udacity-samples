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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 500)

def cnvToFloat(x):
    return float(x.replace(',', ''))

lms = pd.read_csv("./LMS_31JAN2019.csv", keep_default_na=False)
customer = pd.read_csv("./Customers_31JAN2019.csv", keep_default_na=False)
rfdata = pd.read_csv("./RF_Final_Data.csv", keep_default_na=False)
train = pd.read_csv("./train_foreclosure.csv", keep_default_na=False)
test = pd.read_csv("./test_foreclosure.csv", keep_default_na=False)

# print(train.head())

# train['LOAN_AMT'] = lms['LOAN_AMT']
# train['LOAN_AMT'] = train['LOAN_AMT'].astype(float)

# train['LOAN_AMT'] = lms['LOAN_AMT']
# train['LOAN_AMT'] = train['LOAN_AMT'].astype(float)

merged_inner = pd.merge(left=train,right=lms, left_on='AGREEMENTID', right_on='AGREEMENTID')
merged_inner['LOAN_AMT'] = merged_inner['LOAN_AMT'].str.replace(",", "").astype(float)
merged_inner['NET_DISBURSED_AMT'] = merged_inner['NET_DISBURSED_AMT'].str.replace(",", "").astype(float)

merged_inner["CURRENT_ROI"].fillna(0,inplace = True)
merged_inner['CURRENT_ROI'] = merged_inner['CURRENT_ROI'].astype(float)

merged_inner['CURRENT_TENOR'] = pd.to_numeric(merged_inner["CURRENT_TENOR"].str.strip(), downcast='integer').fillna(0)

merged_inner['LAST_RECEIPT_AMOUNT'] = pd.to_numeric(merged_inner['LAST_RECEIPT_AMOUNT'].str.strip(), downcast='float').fillna(0)

merged_inner['BALANCE_TENURE'] = merged_inner['BALANCE_TENURE'].str.replace(",", "")
merged_inner['BALANCE_TENURE'] = pd.to_numeric(merged_inner['BALANCE_TENURE'].str.strip(), downcast='float').fillna(0)

merged_inner['DPD'] = pd.to_numeric(merged_inner['DPD'].str.strip(), downcast='integer').fillna(0)

merged_inner['INTEREST_START_DATE'] = pd.to_datetime(merged_inner['INTEREST_START_DATE']).dt.date
merged_inner['AUTHORIZATIONDATE'] = pd.to_datetime(merged_inner['AUTHORIZATIONDATE']).dt.date
merged_inner['LAST_RECEIPT_DATE'] = pd.to_datetime(merged_inner['LAST_RECEIPT_DATE']).dt.date
merged_inner['LAST_RECEIPT_DATE'] = pd.to_datetime(merged_inner['LAST_RECEIPT_DATE']).dt.date

# merged_inner['MOB'] = pd.to_numeric(merged_inner['MOB'].str.strip(), downcast='integer').fillna(0,inplace = True)

merged_inner['NPA_IN_LAST_MONTH'] = merged_inner['NPA_IN_LAST_MONTH'].apply(lambda x: "NA" if x.strip() == '' else x)
merged_inner['NPA_IN_CURRENT_MONTH'] = merged_inner['NPA_IN_CURRENT_MONTH'].apply(lambda x: "NA" if x.strip() == '' else x)

merged_inner['NPA_IN_LAST_MONTH'] = merged_inner['NPA_IN_LAST_MONTH'].str.upper()
merged_inner['NPA_IN_LAST_MONTH'] = merged_inner['NPA_IN_LAST_MONTH'].str.replace("YES", "0")
merged_inner['NPA_IN_LAST_MONTH'] = merged_inner['NPA_IN_LAST_MONTH'].str.replace("NPA", "0")
merged_inner['NPA_IN_LAST_MONTH'] = merged_inner['NPA_IN_LAST_MONTH'].str.replace("#N/", "1")
merged_inner['NPA_IN_LAST_MONTH'] = merged_inner['NPA_IN_LAST_MONTH'].str.replace("NA", "0")
merged_inner['NPA_IN_LAST_MONTH'] = pd.to_numeric(merged_inner['NPA_IN_LAST_MONTH'].str.strip(), downcast='integer').fillna(0)

merged_inner['NPA_IN_CURRENT_MONTH'] = merged_inner['NPA_IN_CURRENT_MONTH'].str.upper()
merged_inner['NPA_IN_CURRENT_MONTH'] = merged_inner['NPA_IN_CURRENT_MONTH'].str.replace("YES", "0")
merged_inner['NPA_IN_CURRENT_MONTH'] = merged_inner['NPA_IN_CURRENT_MONTH'].str.replace("NPA", "0")
merged_inner['NPA_IN_CURRENT_MONTH'] = merged_inner['NPA_IN_CURRENT_MONTH'].str.replace("#N/", "1")
merged_inner['NPA_IN_CURRENT_MONTH'] = merged_inner['NPA_IN_CURRENT_MONTH'].str.replace("NA", "0")
merged_inner['NPA_IN_CURRENT_MONTH'] = pd.to_numeric(merged_inner['NPA_IN_CURRENT_MONTH'].str.strip(), downcast='integer').fillna(0)

merged_customer = pd.merge(left=merged_inner,right=customer, left_on='CUSTOMERID', right_on='CUSTOMERID')

merged_customer['CUSTOMERID'] = pd.to_numeric(merged_customer['CUSTOMERID'].str.strip(), downcast='integer').fillna(0)
merged_customer['AGE'] = merged_customer['AGE'].apply(lambda x: "0" if x.strip() == '' else x)
merged_customer['AGE'] = merged_customer['AGE'].astype(int)

merged_customer['NO_OF_DEPENDENT'] = merged_customer['NO_OF_DEPENDENT'].apply(lambda x: "0" if x.strip() == '' else x)
merged_customer['NO_OF_DEPENDENT'] = merged_customer['NO_OF_DEPENDENT'].astype(int)

merged_customer['PRE_JOBYEARS'] = merged_customer['PRE_JOBYEARS'].apply(lambda x: "0" if x.strip() == '' else x)
merged_customer['PRE_JOBYEARS'] = merged_customer['PRE_JOBYEARS'].astype(int)

merged_customer['CITY'] = merged_customer['CITY'].apply(lambda x: "NA" if x.strip() == '' else x)
merged_customer['CITY'] = merged_customer['CITY'].str.strip()

merged_customer['QUALIFICATION'] = merged_customer['QUALIFICATION'].apply(lambda x: "NA" if x.strip() == '' else x)

# merged_lms = merged_customer
merged_lms = pd.merge(left=merged_customer,right=rfdata, left_on='AGREEMENTID', right_on='Masked_AgreementID')
# merged_lms = merged_lms.drop(['Preprocessed_EmailBody'],axis=1)
# merged_lms = merged_lms.drop(['Preprocessed_Subject'],axis=1)
# merged_lms = merged_lms.drop(['Preprocessed_Subject'],axis=1)

merged_lms['Date'] = pd.to_datetime(merged_lms['Date']).dt.date

merged_lms['SubType'] = merged_lms['SubType'].apply(lambda x: "NA" if x.strip() == '' else x)
merged_lms['SubType'] = merged_lms['SubType'].str.strip()
merged_lms['SubType'] = merged_lms['SubType'].str.replace(" ", "")
merged_lms['SubType'] = merged_lms['SubType'].str.upper()
merged_lms['Type'] = merged_lms['Type'].apply(lambda x: "NA" if x.strip() == '' else x)
merged_lms['Status'] = merged_lms['Status'].apply(lambda x: "NA" if x.strip() == '' else x)

print(merged_lms.head())
print(merged_lms.dtypes)

merged_lms.to_csv('output.csv', sep=',', encoding='utf-8')

cols = ('QUALIFICATION','CITY','PRODUCT', 'SubType', 'Status', 'Type')
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(merged_lms[c].values)) 
    merged_lms[c] = lbl.transform(list(merged_lms[c].values))

print(merged_lms.head())

cat = len(merged_lms.select_dtypes(include=['object']).columns)
num = len(merged_lms.select_dtypes(include=['int64','float64']).columns)
print('Total Features: ', cat, 'categorical', '+', num, 'numerical', '=', cat+num, 'features')

merged_lms = merged_lms.drop(['EXCESS_AVAILABLE'],axis=1)
merged_lms = merged_lms.drop(['EXCESS_ADJUSTED_AMT'],axis=1)
# merged_lms = merged_lms.drop(['GROSS_INCOME'],axis=1)
merged_lms = merged_lms.drop(['NETTAKEHOMEINCOME'],axis=1)
# merged_lms = merged_lms.drop(['NET_DISBURSED_AMT'],axis=1)
merged_lms = merged_lms.drop(['PRE_EMI_RECEIVED_AMT'],axis=1)
merged_lms = merged_lms.drop(['PRE_EMI_DUEAMT'],axis=1)
merged_lms = merged_lms.drop(['QUALIFICATION'],axis=1)
merged_lms = merged_lms.drop(['AGREEMENTID'],axis=1)
merged_lms = merged_lms.drop(['CUSTOMERID'],axis=1)
merged_lms = merged_lms.drop(['Masked_AgreementID'],axis=1)
merged_lms = merged_lms.drop(['Masked_CustomerID'],axis=1)
merged_lms = merged_lms.drop(['TicketId'],axis=1)
merged_lms = merged_lms.drop(['BRANCH_PINCODE'],axis=1)
# merged_lms = merged_lms.drop(['AGE'],axis=1)

k = 15
corrmat = merged_lms.corr()
colsCorr = corrmat.nlargest(k, 'FORECLOSURE')['FORECLOSURE'].index
cm = np.corrcoef(merged_lms[colsCorr].values.T)
# f, ax = plt.subplots(figsize=(12, 9))
sns.set(font_scale=0.75)
sns.heatmap(cm, cbar=True, annot=True, square=True, vmax=.8, fmt='.2f', yticklabels=colsCorr.values, xticklabels=colsCorr.values)
plt.show()

exit()

train["Total_Stops"] = train["Total_Stops"].str.replace("non-stop", "0 stop")
train["Total_Stops"] = train["Total_Stops"].str.slice(stop=1)
train["Total_Stops"].fillna(0,inplace = True)
# train["Total_Stops"] = train["Total_Stops"].str.replace("stop", "")
# train["Total_Stops"] = pd.to_numeric(train["Total_Stops"].str.strip(), downcast='float')
# train['Total_Stops'] = train['Total_Stops'].astype(int)
train["Total_Stops"] = pd.to_numeric(train["Total_Stops"], downcast='float').fillna(0)

train['Date_of_Journey'] = pd.to_datetime(train['Date_of_Journey']).dt.date

train['Duration'] = pd.to_timedelta(train['Duration'])
train['Dep_Time'] = pd.to_datetime(train['Dep_Time']).dt.time

train['Duration_min'] = train['Duration'].dt.seconds / 60

train.loc[train['Arrival_Time'].str.len() > 6, 'Arrival_Date'] = pd.to_datetime(train['Arrival_Time']).dt.date
train["Arrival_Date"].fillna(train['Date_of_Journey'],inplace = True)

# train['Arrival_Date'] = train['Arrival_Date'].astype(datetime64[D])

train['Arrival_Time'] = pd.to_datetime(train['Arrival_Time']).dt.time

train['Business_class'] = train['Additional_Info'].str.count("Business class")
train['Business_class'] = train['Business_class'].astype(int)

train.loc[train['Additional_Info'].str.count("Long layover")>0, 'Long_layover'] = train["Additional_Info"].str.replace("Long layover", "")
train["Long_layover"] = pd.to_numeric(train["Long_layover"].str.strip())
train["Long_layover"].fillna(0,inplace = True)
train['Long_layover'] = train['Long_layover'].astype(int)

train["In_flight_meal"] = train['Additional_Info'].str.count("In-flight meal not included")
train["In_flight_meal"].fillna(0,inplace = True)
train['In_flight_meal'] = train['In_flight_meal'].astype(int)

train["No_checkin_baggage"] = train['Additional_Info'].str.count("No check-in baggage included")
train["No_checkin_baggage"].fillna(0,inplace = True)
train['No_checkin_baggage'] = train['No_checkin_baggage'].astype(int)

# train['meal'] = train['Additional_Info'].str.count("In-flight meal not included")
# train['baggage'] = train['Additional_Info'].str.count("No check-in baggage included")

train['Arrival_Date'] = train['Arrival_Date'].astype(str)
train['Date_of_Journey'] = train['Date_of_Journey'].astype(str)
train['Arrival_Time'] = train['Arrival_Time'].astype(str)
train['Dep_Time'] = train['Dep_Time'].astype(str)

train.drop(['Additional_Info'],axis=1)
train.drop(['Duration'],axis=1)

train.to_csv('output.csv', sep=',', encoding='utf-8')

# train['Arrival_Date'] = np.where(len(train['Arrival_Time'].str.split(" ")) > 1, train['Arrival_Time'], "")
# print(train['Arrival_Time'].values.str.split(" ").shape)
# (df['First Season'] > 1990).astype(int)

# train['Arrival_Date'] = pd.to_datetime(train['Arrival_Time'])

print(train.head())

print(train.dtypes)

# states = np.unique(np.concatenate((test['Source'].values, test['Destination'].values), axis=0))

# states = np.unique(test['Source'] + test['Destination'])

cols = ('Airline','Date_of_Journey','Source','Destination','Dep_Time','Arrival_Time', 'Arrival_Date')
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))

print(train.head())

df_encoded = train

# exit()

# print(test.head())

# print(train['Price'].describe())

cat = len(train.select_dtypes(include=['object']).columns)
num = len(train.select_dtypes(include=['int64','float64']).columns)
print('Total Features: ', cat, 'categorical', '+', num, 'numerical', '=', cat+num, 'features')

k = 14
corrmat = df_encoded.corr()
cols = corrmat.nlargest(k, 'Price')['Price'].index
cm = np.corrcoef(df_encoded[cols].values.T)
# f, ax = plt.subplots(figsize=(12, 9))
sns.set(font_scale=0.75)
sns.heatmap(cm, cbar=True, annot=True, square=True, vmax=.8, fmt='.2f', yticklabels=cols.values, xticklabels=cols.values)
plt.show()

exit()

# k = 10 #number of variables for heatmap
# cols = corrmat.nlargest(k, 'Price')['Price'].index
# m = np.corrcoef(df_encoded[cols].values.T)
# sns.set(font_scale=0.75)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()

# exit()
# route Source Arivaltime dateofjourny

var = 'Arrival_Time'
data = pd.concat([df_encoded['Price'], df_encoded[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="Price", data=data)
fig.axis(ymin=0, ymax=1900);
plt.show()

exit()

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