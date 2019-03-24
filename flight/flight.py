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

train = pd.read_csv("./Data_Train.csv", keep_default_na=False)
test = pd.read_csv("./Test_set.csv", keep_default_na=False)

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

# train.to_csv('output.csv', sep=',', encoding='utf-8')

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

k = 6
corrmat = df_encoded.corr()
cols = corrmat.nlargest(k, 'Price')['Price'].index
cm = np.corrcoef(df_encoded[cols].values.T)
# f, ax = plt.subplots(figsize=(12, 9))
sns.set(font_scale=0.75)
sns.heatmap(cm, cbar=True, annot=True, square=True, vmax=.8, fmt='.2f', yticklabels=cols.values, xticklabels=cols.values)
plt.show()

exit()

# change all the column of orients to StandardScaler
for col in train.columns:
    if 'orient' in col:
        scaler = StandardScaler()
        train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1))
        test[col] = scaler.fit_transform(test[col].values.reshape(-1, 1))


