import pandas as pd
import numpy as np
df = pd.read_csv('development_sample.csv', header=0, index_col=None,low_memory=False)
dftest = pd.read_csv('assessment_sample.csv', header=0, index_col=None,low_memory=False)
#catColumns = df.select_dtypes(include=['object']).columns
#df.loc[:,catColumns]= df[catColumns].astype('object')
df.dropna(subset=['TARGET'],inplace=True)
dfeature = df.drop( ['TARGET','ID2'],axis=1).copy()
dfeature_test = dftest.drop( ['ID2'],axis=1).copy()
#df.head()

# hack the data types and unique values
pd.set_option('display.max_rows', 500)
desc = pd.DataFrame()
desc['dtypes'] = dfeature.dtypes
desc['nuniques'] = dfeature.nunique()
desc['count'] = dfeature.count()
desc['value set'] = dfeature.apply(lambda x:set(x), axis=0)
desc.to_csv('develop_desc.csv')

# hack the test data set
desc = pd.DataFrame()
desc['dtypes'] = dfeature_test.dtypes
desc['nuniques'] = dfeature_test.nunique()
desc['count'] = dfeature_test.count()
desc['value set'] = dfeature_test.apply(lambda x:set(x), axis=0)
desc.to_csv('assessment_desc.csv')

# explore the data with hist and pdf plot
import matplotlib.pyplot as plt
for col in dfeature.columns[0:]:
    try:
        if dfeature.dtypes[col] != dfeature.dtypes['HOUR']:
            plt.figure()
            plt.hist(dfeature[col])
            plt.title(col)
        else:
            ser = dfeature[col]
            sergp = ser.groupby(ser.values).size().sort_values(ascending=False)
            plt.figure()
            plt.plot(sergp.values)
            plt.title(col)
            plt.xticks(range(len(sergp)), sergp.index, rotation='vertical')
    except ValueError:
            print(col)
plt.hist(dfeature.COMPFIELD.dropna())
plt.title('COMPFIELD')
plt.show()

# try to convert some ordinal catetorical variables to numerical
def str2float(x):
    #try:
#       try:
#           if np.isnan(x):
#               return x
#       except TypeError:
#               pass
        if isinstance(x,float) or isinstance(x,int):
            return x
        if 'low-' in x:
            return (float(x.split('low-')[1]) - 1) * 0.5
        if '>=' in x:
            return (float(x.split('>=')[1]) + 1) * 1.5
        if '<=' in x:
            return (float(x.split('<=')[1]) - 1) * 0.5
        if '>' in x:
            return (float(x.split('>')[1]) + 1) * 1.5
        if '<' in x:
            return (float(x.split('<')[1]) - 1) * 0.5
        if '-' in x:
            if x.split('-')[1] == '':
                return np.nan
            else:
                return ( float(x.split('-')[0]) + float(x.split('-')[1]) )* 0.5
        if 'N\\A' in x:
                return np.nan
        return float(x)
    #except ValueError:
    #    print(x)
dfnew = pd.DataFrame()
for col in dfeature.select_dtypes(include=['object']).columns:
    #print(col)
    try:
        dfeature.loc[:, col+'numerical'] = dfeature[col].apply(str2float)
        dfeature_test.loc[:, col+'numerical'] = dfeature_test[col].apply(str2float)
        #dfeature.loc[:, col] = dfeature[col].apply(str2float)
        #dfeature_test.loc[:, col] = dfeature_test[col].apply(str2float)
    except ValueError:
        print('Error',col)
dfeature.head()

#fill nan values, different strategies for different data types: median for float, mode for integer, 'Missing' for categorical
def fill_nan(df):
    fillFloatColumns = df.select_dtypes(include=['float']).median()
    fillIntColumns = df.select_dtypes(include=['int']).mode().iloc[0]
    fillCatColumns = pd.Series( 'Missing', index=df.select_dtypes(include=['object']).columns)
    fillValue = fillFloatColumns.append(fillIntColumns).append(fillCatColumns)
    #fillValue
    df = df.fillna(value=fillValue)
    return df

dfeature      = fill_nan(dfeature)
dfeature_test = fill_nan(dfeature_test)

dfcorr = dfeature.corr()
colinearList = []
for idx in dfcorr.index:
    for col in dfcorr.columns:
        if dfcorr.loc[idx,col] > 0.9 and idx!=col:
            colinearList.append( [set([idx,col]) , dfcorr.loc[idx,col]] )
(colinearList)

# feature engineering as a transformer to avoid touching the TARGET information while calculate woe
from sklearn.base import BaseEstimator
import time
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler,QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile, f_classif,SelectFromModel

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import category_encoders as ce
from tempfile import mkdtemp


def lap_div(x,y):
    #laplace division in case 0/0
    return (x+1)/(y+1)
class derivesFeatureGenerator(BaseEstimator):
    #calculate woe
    def __init__(self):
        self.woeDict = {}
    def fit(self, X, y):
        for col in X.select_dtypes(include=['object']).columns:
            xtab = pd.crosstab(X[col],y)
            xtab['woe'] = np.log(  lap_div( xtab[1],xtab[1].sum()) /  lap_div(xtab[0],xtab[0].sum()) )
            self.woeDict[col] = xtab['woe'].to_dict()
        return self
    def transform(self, X, y=None):
        for col in X.select_dtypes(include=['object']).columns:
            mapper = self.woeDict[col]
            nafill = np.median(list(mapper.values()))
            X[col+'_woe'] = X[col].map( lambda x: mapper.setdefault(x, nafill) )
        return X
class clusterTransformer(BaseEstimator):
    #try to use the leaf id from xgboost or randomForest as  inputs of final estimator, make a chain of xgboost/RF + LR/SVM
    def __init__(self,estimator):
        self.clf = estimator
    def fit(self, X, y):
        self.clf.fit(X,y)
        return self
    def transform(self, X, y=None):
        Xleaf = self.clf.apply(X) 
        return Xleaf
class ensembleModel(BaseEstimator):
    # use a emsemble Model as the final estimator
    def __init__(self,estimators=[]):
        self.clfs = estimators
    def fit(self, X, y):
        for clf in self.clfs:
            print('training', type(clf),datetime.now().strftime('%H:%M:%S'))
            clf.fit(X,y)
        return self
    def predict(self, X, y=None):
        ypre = pd.DataFrame()
        mid = 0
        for clf in self.clfs:
            print('predicting', type(clf),datetime.now().strftime('%H:%M:%S'))
            ypre[mid] = clf.predict(X)
            mid += 1
        return ypre
    def predict_proba(self, X, y=None):
        yproba = pd.DataFrame()
        mid = 0
        for clf in self.clfs:
            print('predicting_proba', type(clf),datetime.now().strftime('%H:%M:%S'))
            yproba[mid] = clf.predict_proba(X)[:,1]
            mid += 1
        return yproba
    def transform(self, X, y=None):
        yproba = pd.DataFrame()
        mid = 0
        for clf in self.clfs:
            yproba[mid] = clf.predict_proba(X)[:,1]
            mid += 1
        return yproba
class ensembleWithStrategy(BaseEstimator):
    # use max or min to vote
    def __init__(self,estimators=[],operator=None):
        self.clfs = estimators
        self.opt = operator
    def fit(self, X, y):
        for clf in self.clfs:
            print('training', type(clf),datetime.now().strftime('%H:%M:%S'))
            clf.fit(X,y)
        return self
    def predict(self, X, y=None):
        ypre = pd.DataFrame()
        mid = 0
        for clf in self.clfs:
            print('predicting', type(clf),datetime.now().strftime('%H:%M:%S'))
            ypre[mid] = clf.predict(X)
            mid += 1
        return ypre.apply(self.opt, axis=1)
    def predict_proba(self, X, y=None):
        yproba = pd.DataFrame()
        mid = 0
        for clf in self.clfs:
            print('predicting_proba', type(clf),datetime.now().strftime('%H:%M:%S'))
            yproba[mid] = clf.predict_proba(X)[:,1]
            mid += 1
        return yproba.apply(self.opt, axis=1)

cachedir = mkdtemp()


derivs = derivesFeatureGenerator()
ohc = ce.OneHotEncoder(handle_unknown='ignore')
mms = MinMaxScaler()
sts = StandardScaler()
qts = QuantileTransformer()

sfv = SelectPercentile(f_classif, percentile=90)
sfm = SelectFromModel(ExtraTreesClassifier(n_estimators=960, max_depth=10, class_weight='balanced',
                      max_features=0.3, min_samples_split=1000, min_samples_leaf=500, random_state=0, n_jobs=-1)
                     )
pca = PCA(n_components=0.7)
xgc = XGBClassifier(n_estimators=320*2,  nthread=16, min_child_weight=200 ,max_depth=3,
                    subsample=0.9, colsample_bytree=0.9, scale_pos_weight=1.0 )
lr = LogisticRegression( C=100.0, class_weight='balanced', max_iter=1000, penalty='l2')
knc = KNeighborsClassifier( n_neighbors=3, n_jobs=-1)
gnc = GaussianNB()
svc = SVC( C=1, probability=True,class_weight='balanced')
mlp = MLPClassifier(alpha=1)

emm = ensembleModel(estimators=[xgc,lr,mlp])
ews = ensembleWithStrategy(estimators=[xgc,lr,mlp],operator=max)
lrem = LogisticRegression(  class_weight='balanced', max_iter=1000, penalty='l2')

cluster = clusterTransformer(xgc)
enc = OneHotEncoder()

rfc = RandomForestClassifier(n_estimators=9600, max_depth=10, class_weight='balanced',
                             criterion='entropy',max_features=0.9,min_samples_split=50,
                             min_samples_leaf=20,random_state=0, n_jobs=-1)
pipe = Pipeline([('derivesFeature',derivs),
                 ('oneHotEncoding', ohc),
                 ('numericalScales', qts),
                 ('featureSelection1',sfv), 
                 #('featureSelection2',sfm),
                 #('featureDecompositon',pca),
                 ('clf',xgc),
                 
                 #('enemble',emm),
                 #('lrem',lrem),
                 
                 #try the combination of xgboost+LR
                 #('cluster',cluster),
                 #('one hot encoder',enc),
                 #('logisticR', lr),
                 
                ])
