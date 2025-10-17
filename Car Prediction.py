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
def dot (xi, w):
    n = len(xi)
    result = 0.0
    for j in range(n):
        result += xi[j] * w[j]
    return result
print(dot(xi, w))

def linear_regression(xi):
    return w0 + dot(xi, w)
print(linear_regression(xi))

#Short Notation
w_new = [w0] + w
print(w_new)

def linear_regression(xi):
    x1 = [1] + xi
    return dot(x1, w_new)
print(linear_regression(xi))

#Making predictions for multiple cars
xi = [185, 17, 1385]
w0 = 7.17
w = [0.01, 0.04, 0.002]

x1 = [1, 148, 24, 1385]
x2 = [1, 132, 25, 2031]
x10= [1, 453, 11, 853]

X = [x1, x2, x10]
X = np.array(X)
print(X)

def linear_regression(X):
    return X.dot(w_new)
print(linear_regression(X))


#Training a linear regression model
def train_linear_regression(X, y):
    pass

X = [
    [148, 54, 1385],
    [132, 55, 2031],
    [453, 11, 853],
    [132, 25, 2031],
    [453, 71, 853],
    [53, 18, 853],
    [32, 25, 2031],
    [22, 11, 853],
    [57, 32, 853],
]

X = np.array(X)
print(X)

#Gram matrix
ones = np.ones(X.shape[0])
print(ones)

X = np.column_stack([ones, X])
print(X)

y = [7.4, 7.12, 6.80, 7.12, 6.80, 7.50, 7.25, 7.60, 7.45]
XTX = X.T.dot(X)
print(XTX)
XTX_inv = np.linalg.inv(XTX)
XTX.dot(XTX_inv).round(1)
print(XTX.dot(XTX_inv).round(1))

w_full = XTX_inv.dot(X.T).dot(y)
print(w_full)

w0 = w_full[0]
w = w_full[1:]
print(w0, w)

"""
def train_linear_regression(X, y):
       Train linear regression using the normal equation.

        This implementation is robust to a singular X^T X matrix. It first
        attempts to compute the inverse; if that fails because the matrix is
        singular, it falls back to the Moore-Penrose pseudo-inverse.

        Inputs:
            - X: 2D numpy array of shape (n_samples, n_features)
            - y: 1D array-like of length n_samples

        Returns:
            - w0: intercept (float)
            - w: 1D numpy array of feature weights
      
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        try:
                XTX_inv = np.linalg.inv(XTX)
        except np.linalg.LinAlgError:
                # XTX is singular (not invertible). Use pseudo-inverse instead.
                XTX_inv = np.linalg.pinv(XTX)

        w_full = XTX_inv.dot(X.T).dot(y)

        return w_full[0], w_full[1:]
print(train_linear_regression(X, y))

"""



#Car Price Baseline Model, this is just a placeholder for further model development

df_train.dtypes #Check the data types of the training dataframe
print(df_train.dtypes)

df_train.columns
print(df_train.columns)

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

X_train = df_train[base].values
df_train = df_train[base].fillna(0).values       #Handling missing values by filling them with 0
print(df_train)

train_linear_regression(X_train, y_train)

"""
y_pred = w0 + X_train.dot(w) #Making predictions on the training set
print(y_pred)

sns.histplot(y_pred)
sns.histplot(y_train)
plt.show()
"""
