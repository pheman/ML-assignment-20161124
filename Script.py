import rediscluster as rdc
REDIS_HOSTS = [
    {'host': 'xxx', 'port': '6379'},
]

redis_conn = rdc.StrictRedisCluster(startup_nodes=REDIS_HOSTS, decode_responses=False)

def get_dummies_groupby(df, factor_dict):
    
    df_std_columns = [dim +'='+ factor for dim in factor_dict.keys() for factor in factor_dict[dim]]
    df = pd.get_dummies(df, prefix_sep='=', sparse=False)
    df_get_columns= df.columns
    df_more_columns = set(df_std_columns) - set(df_get_columns)
    for col in df_more_columns:
        df[col] = pd.Series([0.0]*len(df),index=df.index)
        
    df = df.groupby(df.index).max()  #how to handle the sorted list

    df.sort_index(axis=1, inplace=True)        # sort the columns  
    dfcsr = csr_matrix(df.values)#.tocsr()

    return dfcsr,df.index,list(df.columns)
   
def get_dummies_groupby_mp(df, factor_dict, n_jobs):
    npart = n_jobs * 6
    ngap = int(len(df)/npart)
    pool = Pool(processes=n_jobs)
    dfinputList = []
    for i in range(npart):
        dfpart = df.iloc[i*ngap : (i+1)*ngap]
        if i == npart -1:
            dfpart = df.iloc[i*ngap : ]
        dfinputList.append(dfpart)
    resList = pool.starmap(get_dummies_groupby, [(dfpart, factor_dict ) for dfpart in dfinputList] )
    pool.close()
    pool.join()
    
    dfresList = [ x[0] for x in resList]
    idxList = pd.DataFrame().index
    for x in resList:
        idxList  = idxList.append(x[1])
    
    dfres = vstack(dfresList)  ### consider duplication
    #dfres = pd.SparseDataFrame(dfres, index = idxList, columns = resList[0][2],default_fill_value=0)
    #dfres = dfres.groupby(dfres.index).max()  #how to handle the sorted list
    return dfres,idxList,resList[0][2]


amountByEmployee      = df[['a','b','c']].drop_duplicates().groupby('a').agg(
                                {'b':[np.mean,np.sum], 'c':len})
                                
redis_conn.set(name,vpickle)
