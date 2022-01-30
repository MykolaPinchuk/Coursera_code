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
print(hspr.head())
print(hspr.shape)
  

#%% data cleaning ###

hspr0 = hspr.copy()
cols_tokeep = ['SalePrice', 'LotArea', 'Neighborhood', 'OverallQual', 'OverallCond', 'YearBuilt', 'ExterCond',
               'BsmtFinSF1', 'TotalBsmtSF', 'HeatingQC', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 
               'KitchenQual', 'GarageArea', 'GarageCars', 'TotRmsAbvGrd', 'BedroomAbvGr',
               'ExterQual']
hspr = hspr[cols_tokeep]
hspr.info()
# there are no missing values.

ord_cols = ['ExterCond', 'HeatingQC', 'KitchenQual', 'ExterQual']
hspr[ord_cols] = hspr[ord_cols].replace(['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1,2,3,4,5])
#print(hspr.BsmtCond.value_counts())

# it makes sense to replace YearBuilt with Age
hspr['Age']=2010-hspr.YearBuilt
hspr.drop(columns=['YearBuilt'], inplace=True)

#sns.boxplot(x='MSSubClass', y='SalePrice', data=hspr0)


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

#%% Encode categorical variables ###

nbh_counts = hspr.Neighborhood.value_counts()
other_nbhs = list(nbh_counts[nbh_counts < 30].index)
hspr['Neighborhood'].replace(other_nbhs, 'Other', inplace=True)

hspr.drop(columns=['Neighborhood'],inplace=True)



#%% Model fitting ###

y = hspr.SalePrice
X = hspr.drop(columns=['SalePrice'])
X_train = X.copy()

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

test_s = pd.read_csv("test.csv") 
print(test_s.head())
print(test_s.shape)

id_ = test_s.Id
cols_tokeep.remove('SalePrice')
test_s = test_s[cols_tokeep]

ord_cols = ['ExterCond', 'HeatingQC', 'KitchenQual', 'ExterQual']
test_s[ord_cols] = test_s[ord_cols].replace(['Po', 'Fa', 'TA', 'Gd', 'Ex'], [1,2,3,4,5])
test_s['Age']=2010-test_s.YearBuilt
test_s.drop(columns=['YearBuilt'], inplace=True)
test_s['LotArea']=np.log1p(test_s.LotArea)

test_s.drop(columns=['Neighborhood'],inplace=True)

#%% predict ###

X = test_s.copy()

X.loc[X.GarageArea.isna(),'GarageArea']=0
X.loc[X.GarageCars.isna(),'GarageCars']=0
X.loc[X.BsmtFinSF1.isna(),'BsmtFinSF1']=0
X.loc[X.TotalBsmtSF.isna(),'TotalBsmtSF']=0
X.loc[X.KitchenQual.isna(),'KitchenQual']=2

X = s.transform(X)

yhat = fullmodel.predict(X)

results = pd.DataFrame({'Id': id_, 'SalePrice': yhat}, columns=['Id', 'SalePrice'])
results.to_csv('HousePrices_subm5_2.csv', index=False)

#X_train.columns == test_s.columns


# taking log of y somehow crews things up: score goes from 0.34 to 0.1488...
# this is difference btw 5_1 and 5_2. I am confused...

# 