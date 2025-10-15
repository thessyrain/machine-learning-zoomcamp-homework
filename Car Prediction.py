#Machine Learning for Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



#Data Preparation 
data = 'https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/data.csv?raw=true'

df = pd.read_csv(data)

#Exploratory Data Analysis (EDA)
print(df.head())
df.columns = df.columns.str.lower().str.replace(' ', '_')
print(df.columns)

# Check for types of values for the indexed columns
string = df.dtypes[df.dtypes == 'object'].index.tolist()
print(string)

for col in string:
    df[col] = df[col].str.lower().str.replace(' ', '_')
    print(df.shape)
    print(df.head())


#Exploratory Data Analysis (EDA)
for col in df.columns:
    print(col)
    print(df[col].unique() [:5])
    print(df[col].nunique() )
    print()


#Distribution of price MRSP
sns.histplot(df.msrp, bins=50)
plt.xlabel('msrp')
plt.ylabel('count')
plt.title('Distribution of msrp')
plt.savefig("msrp_distribution.png")
plt.show()

#Distribution of price MRSP<100,000
sns.histplot(df.msrp [df.msrp < 100000], bins=50)
plt.xlabel('msrp')
plt.ylabel('count')
plt.title('Distribution of msrp < 100,000')
plt.savefig("msrp_distribution_100k.png")
plt.show()

#Applying the log distribution to the price MRSP
np.log1p([0, 1, 10, 100, 1000, 10000])
price_logs = np.log1p(df.msrp[df.msrp < 100000])
price_logs = np.log1p(df.msrp)
print(df.msrp.min())
print(price_logs)


#Normal distribution of log(msrp)
sns.histplot(price_logs, bins=50)
plt.xlabel('log(msrp)')
plt.ylabel('count')
plt.title('Distribution of log(msrp)')
plt.savefig("log_msrp_distribution.png")
plt.show()

#Missing values
print(df.isnull().sum())
print(df.isnull().mean())
print(df.shape)

#Setting up the validation framework
n = len(df)
n_val = int(len(df) * 0.2)
n_test = int(len(df) * 0.2)
n_train = n - n_val - n_test
print(n, n_train, n_val, n_test)

#Shuffle the records in data frame
df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train + n_val]
df_test = df.iloc[n_train + n_val:]
print(df_train, df_val, df_test)

idx = np.arange(n)
np.random.seed(2)
np.random.shuffle(idx)
df = df.iloc[idx]
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train : n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val :]]
print(df_train, df_val, df_test)

#Check the lengths of the dataframes
len(df_train), len(df_val), len(df_test)
print(len(df_train), len(df_val), len(df_test))

#Reset the indices
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
print(df_train, df_val, df_test)

#Getting the target variable

y_train = np.log1p(df_train.msrp.values) #Applying the log transformation to the target variable
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)
print(y_train, y_val, y_test)

#Delete the target variable from the dataframes
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']
print(df_train, df_val, df_test)

len(y_train)
print(len(y_train))



#Linear Regression
df_train.iloc[10]
print(df_train.iloc[10])
xi = [185, 17, 1385]
w0 = 7.17
w = [0.01, 0.04, 0.002]

def linear_regression(xi):
    n = len(xi)
    pred = w0
    for j in range(n):
        pred += w[j] * xi[j]
    return pred
print(linear_regression(xi))


#getting the exponential of the prediction
np.expm1(12.469999999999999)
print(np.expm1(12.469999999999999)) #inverse of log1p

#log1p of the prediction
np.log1p (260405.72118870862)
print(np.log1p (260405.72118870862)) #log1p of the prediction


#Linear Regression  of vector form

