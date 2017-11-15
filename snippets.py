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
