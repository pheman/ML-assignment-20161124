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
