"""
Created on Sat Jan 29 12:34:42 2022
"""

import numpy as np
import pickle, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import boxcox
from scipy.special import inv_boxcox

pd.set_option('display.max_columns', 35)

os.chdir("H:/Dropbox/Kaggle/house_prices")

hspr = pd.read_csv("train.csv") # titanic_fullsample
hspr['sample']='train'
test_s = pd.read_csv("test.csv") 
test_s['SalePrice']=np.nan
test_s[['SalePrice', 'sample']] = [np.nan, 'test']
hspr=pd.concat([hspr, test_s])
hspr.reset_index(inplace=True, drop=True)
print(hspr.head())
print(hspr.shape)

#%% data cleaning ###

hspr0 = hspr.copy()
cols_tokeep = ['SalePrice', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'ExterCond',
               'BsmtFinSF1', 'TotalBsmtSF', 'HeatingQC', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 
               'KitchenQual', 'GarageArea', 'GarageCars', 'TotRmsAbvGrd', 'BedroomAbvGr',
               'ExterQual', 'sample']
hspr = hspr[cols_tokeep]
hspr.dropna(subset=hspr.columns.drop('SalePrice'), inplace=True)
hspr.info()
# there are no missing values.

ord_cols = ['ExterCond', 'HeatingQC', 'KitchenQual', 'ExterQual']
hspr[ord_cols] = hspr[ord_cols].replace(['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1,2,3,4,5])
#print(hspr.BsmtCond.value_counts())

# it makes sense to replace YearBuilt with Age
hspr['Age']=2010-hspr.YearBuilt
hspr.drop(columns=['YearBuilt'], inplace=True)

#%% check for skew and outliers ###

# check skew:
    
temp = (hspr.dtypes == np.int64)
num_cols = hspr.columns[temp]
skew_vals = hspr[num_cols].skew() 
skew_limit = 1
    
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))

print(skew_cols)

# transform LotArea
hspr['LotArea']=np.log1p(hspr.LotArea)
hspr['LotArea'].skew()

#%% Model fitting ###

X = hspr[hspr['sample']=='train'].copy()
X.drop(columns=['sample', 'SalePrice'], inplace=True)
y = hspr.SalePrice[hspr['sample']=='train'].copy()

s = StandardScaler()
X = s.fit_transform(X)

# first, fit ols #

lm = LinearRegression()
predictions = cross_val_predict(lm, X, y, cv = 10)
score_lm = r2_score(y, predictions)
# 79.4%

# second, try lasso #

lm = LinearRegression()
predictions = cross_val_predict(lm, X, y, cv = 10)
score_lm = r2_score(y, predictions)
# 79.4%



s = StandardScaler()
alphas = np.geomspace(1e-5, 1e-1, num=19)
scores = []
coefs = []
for alpha in alphas:
    las = Lasso(alpha=alpha, max_iter=100000)
    
    estimator = Pipeline([
        ("scaler", s),
        ("lasso_regression", las)])

    predictions = cross_val_predict(estimator, X, y, cv = 10)
    score = r2_score(y, predictions)
    scores.append(score)
    
list(zip(alphas,scores))

plt.figure(figsize=(10,6))
plt.semilogx(alphas, scores, '-o')
plt.xlabel('$\\alpha$')
plt.ylabel('$R^2$');

alpha_opt=0.002

fullmodel = Lasso(alpha=alpha_opt)
X = s.fit_transform(X)
fullmodel.fit(X,y)

X_train_s = X.copy()


# %% test sample ###



#%% predict ###


X = s.transform(X)

yhat = fullmodel.predict(X)

results = pd.DataFrame({'Id': id_, 'SalePrice': yhat}, columns=['Id', 'SalePrice'])
results.to_csv('HousePrices_subm5_2.csv', index=False)

#X_train.columns == test_s.columns


# taking log of y somehow crews things up: score goes from 0.34 to 0.1488...
# this is difference btw 5_1 and 5_2. I am confused...

# 