#!/bin/sh

source ~/.bashrc

fileid=1345
base_dir=hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000
async_dir=${base_dir}"/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/async"

while true
#for i in {0..4}
do  
    hadoop fs -test -e ${async_dir}"/url_pv/_SUCCESS"
    if [ $? -eq 0 ]; then
        hadoop fs -test -e ${async_dir}"/url_save/url_list_"${fileid}"/_SUCCESS"
        if [ $? -eq 0 ]; then
             ((fileid=fileid+1))
             echo file existed, fileid=${fileid}
        else 
             echo sparksubmit select_url_list.py --fileid ${fileid}
             sparksubmit select_url_list.py --fileid ${fileid}
        fi
    else
        sleep 300
        echo waiting pv file
    fi
    
done

# -*- coding: utf-8 -*-
"""
Created on 2017-12-06
get uid-url 

@author: zhangruibin
"""

from random import random
from operator import add
import subprocess, sys, os, time 
import json
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, HiveContext, Row
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark import StorageLevel
from optparse import OptionParser
import datetime
import time
import hashlib
from py4j.protocol import Py4JJavaError

def sum_df(df, col):
    sumcol  = df.agg(F.sum(col).alias('sum')).collect()[0]['sum']
    return sumcol
def parse3(line):
    linesp = line.split('\t')
    try:
        return (linesp[0], linesp[1], linesp[2])
    except:
        return (linesp[0], '', '')
def get_host_list():
    f = open('hostFrom')
    lines = f.readlines()
    hostList = []
    for line in lines:
        hostList.append( line.split('__')[-1][0:-1] )    
    return  set(hostList)


def get_url(line):
    line_split = line.split('\t')
    uid  = line_split[0].split('\x01')[0]
    numItems = len(line_split) - 1 
    try:
        date = line_split[0].split('\x01')[1]
    except IndexError:
        date = 'null'
    
    urlList = [ url.split('\x011')[0].split('\x012')[0] for url in line_split[1:] ]
    
    host_url, query, other = [], '', ''
    for url in urlList:
        if   url[0] == 'u':
            validurl  = url[1:].split('\x01')[0]
            validhost = validurl.split('http://')[1].split('/')[0]
            host_url.append( (validhost,validurl) )
    
    uid_host_url_pair = [(uid, host_url_pair[0], host_url_pair[1], 
                   hashlib.md5(host_url_pair[1]).hexdigest()) for host_url_pair in host_url]
    
    #return Row(uid = uid, date=date, numItems=numItems, http=http, query=query, other=other, urlList=validList)
    return uid_host_url_pair


def main(fileid):
    base_dir = 'hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000'
    async_dir = base_dir + '/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/async'

    date = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y%m%d")
    log_path = base_dir + '/home/nlp/offline/right_recom/safebox/nlp_correlation/{date}/0000/query_coocur/part-*'.format(date=date)
    print log_path
    urllog = sc.textFile(log_path)
    hostList = get_host_list()
    hostList_bd = sc.broadcast(hostList)
    urllog = urllog.flatMap(get_url)

    uid_host_url_all = sq.createDataFrame(urllog, ['uid','host','url','urlhash'])
    uid_host_url_filter = uid_host_url_all.filter(uid_host_url_all.host.isin(hostList_bd.value))
    url_group = uid_host_url_filter.groupby('url').agg(F.count(uid_host_url_filter.url).alias('pv'))
    url_pv = url_group.select('url', 'pv')
    
    filtered_url_pv = url_pv.filter("pv > 5")#.repartition(1000)#.sortBy(lambda x: x[1], ascending=False)

    unsorted_updf = filtered_url_pv.persist()#.toDF(['url','pv'])
    SelectNewUrl = True
    if SelectNewUrl:
    	try:
        	utt_path = base_dir + '/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/async/url_title_tag/utt_*/part-*'
        	utt = sc.textFile(utt_path).map(parse3).toDF(['url','title','tag'])
        	utt = utt.filter(utt.tag != '').dropDuplicates()

        	before_join_num = unsorted_updf.count()
        	joined_updf     = unsorted_updf.join(utt, 'url', 'left')
        	#after_join_num = joined_updf.count()
        	print 'before and after join ', before_join_num # after_join_num
        	new_url_updf = joined_updf.filter(F.isnull('tag'))
        	new_url_rdd = new_url_updf.map(lambda x: (x['url'], x['pv']))
                new_num = new_url_rdd.count()
                print 'total new url ', new_num
    	except Py4JJavaError:
        	new_url_rdd = unsorted_updf.map(lambda x: (x['url'], x['pv']))

    new_url_rdd = unsorted_updf.map(lambda x: (x['url'], x['pv']))
    sample_num = 500000
    new_sort_url = new_url_rdd.sortBy(lambda x: x[1], ascending=False).toDF(['url','pv'])
    url_list = new_sort_url.limit(sample_num)

#    new_url_length = new_url_rdd.count()
#    url_list = new_url_rdd.toDF(['url','pv']).sample( False, min( 0.9, 1.0*sample_num/new_url_length)   ).persist()
#    print 'sample rate is ', sample_num/new_url_length

    url_list.show(truncate=False)
    url_list = url_list.map(lambda x: x['url'])#.repartition(100) # strange code \x01 

    save_url_list = async_dir + '/url_save/url_list_{0}'.format(fileid)
    os.system('hadoop fs -rmr {sdir}'.format( sdir=save_url_list ) )
    url_list.saveAsTextFile( save_url_list , "org.apache.hadoop.io.compress.GzipCodec")


if __name__ == "__main__":
    usage = "usage: %prog [options] arg" 
    parser = OptionParser(usage = usage)
    parser.add_option("--fileid", dest='fileid',  help='fileid')
    parser.add_option("--date", dest='date',  help='date')
    (options, args) = parser.parse_args()
    conf = SparkConf().setAppName("zhangruibin_select_url_list") \
                      .set("spark.driver.memory", "500g") \
                      .set("spark.driver.maxResultSize", "5g") \
                      .set("spark.executor.memory", "7g") \
                      .set("spark.storage.memoryFraction", 0.6) \
                      .set("spark.shuffle.memoryFraction", 0.3) \
                      .set("spark.network.timeout", 1200) \
                      .set("spark.executor.instances", 500) \
                      .set("spark.executor.cores", 2) \
                      .set("spark.default.parallelism", 4000) \
                      .set("spark.yarn.priority", "VERY_HIGH") \
                      .set("spark.shuffle.service.enabled", True) \
                      .set("spark.yarn.queue", "root.hdp-reader") \
                      .set("spark.akka.frameSize", 250)

# root.hdp-reader
    sc = SparkContext(conf=conf)
    sq = SQLContext(sc)
    sq = HiveContext(sc)
    main(options.fileid)

    sc.stop()


# -*- coding: utf-8 -*-
"""
Created on 2017-12-06
get uid-url 

@author: zhangruibin
"""

from random import random
from operator import add
import subprocess, sys, os, time 
import json
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, HiveContext, Row
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark import StorageLevel
from optparse import OptionParser
import datetime
import time
import hashlib
from py4j.protocol import Py4JJavaError

def sum_df(df, col):
    sumcol  = df.agg(F.sum(col).alias('sum')).collect()[0]['sum']
    return sumcol
def parse2(line):
    linesp = line.split('\t')
    try:
        return (linesp[0], int(linesp[1]))
    except:
        return (linesp[0], 0)
def parse3(line):
    linesp = line.split('\t')
    try:
        return (linesp[0], linesp[1], linesp[2])
    except:
        return (linesp[0], '', '')

def main(fileid):
    base_dir = 'hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000'
    async_dir = base_dir + '/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/async'

    url_pv       = sc.textFile( async_dir + '/url_pv/part-*').map(parse2)
    filtered_url_pv = url_pv.filter( lambda x: x[1]>30)#.repartition(1000)#.sortBy(lambda x: x[1], ascending=False)

    unsorted_updf = filtered_url_pv.toDF(['url','pv'])
    if False:
    	try:
        	utt_path = base_dir + '/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/async/url_title_tag/utt_*/part-*'
        	utt = sc.textFile(utt_path).map(parse3).toDF(['url','title','tag'])
        	utt = utt.filter(utt.tag != '').dropDuplicates()

        	before_join_num = unsorted_updf.count()
        	joined_updf     = unsorted_updf.join(utt, 'url', 'left')
        	after_join_num = joined_updf.count()
        	print 'before and after corss ', before_join_num, after_join_num, 1000000
        	new_url_updf = joined_updf.filter(F.isnull('tag'))
        	new_url_rdd = new_url_updf.map(lambda x: (x['url'], x['pv']))
    	except Py4JJavaError:
        	new_url_rdd = unsorted_updf.map(lambda x: (x['url'], x['pv']))

    new_url_rdd = unsorted_updf.map(lambda x: (x['url'], x['pv']))
#    new_sort_url = new_url_rdd.sortBy(lambda x: x[1], ascending=False).toDF(['url','pv'])
#    url_list = new_sort_url.limit(1000000).persist()
    new_url_length = new_url_rdd.count()
    sample_num = 500000.0
    url_list = new_url_rdd.toDF(['url','pv']).sample( False, min( 0.9, sample_num/new_url_length)   ).persist()
    print 'sample rate is ', sample_num/new_url_length
    url_list.show(truncate=False)
    url_list = url_list.map(lambda x: x['url'])#.repartition(100) # strange code \x01 
    save_url_list = async_dir + '/url_save/url_list_{0}'.format(fileid)
    os.system('hadoop fs -rmr {sdir}'.format( sdir=save_url_list ) )
    for item in url_list.take(10):
        print item
    url_list.saveAsTextFile( save_url_list , "org.apache.hadoop.io.compress.GzipCodec")


if __name__ == "__main__":
    usage = "usage: %prog [options] arg" 
    parser = OptionParser(usage = usage)
    parser.add_option("--fileid", dest='fileid',  help='fileid')
    (options, args) = parser.parse_args()
    conf = SparkConf().setAppName("zhangruibin_select_url_list") \
                      .set("spark.driver.memory", "500g") \
                      .set("spark.driver.maxResultSize", "5g") \
                      .set("spark.executor.memory", "7g") \
                      .set("spark.storage.memoryFraction", 0.1) \
                      .set("spark.shuffle.memoryFraction", 0.5) \
                      .set("spark.network.timeout", 1200) \
                      .set("spark.executor.instances", 500) \
                      .set("spark.executor.cores", 1) \
                      .set("spark.default.parallelism", 4000) \
                      .set("spark.yarn.priority", "VERY_HIGH") \
                      .set("spark.shuffle.service.enabled", True) \
                      .set("spark.yarn.queue", "root.hdp-reader") \
                       .set("spark.akka.frameSize", 250)

# root.hdp-reader
    sc = SparkContext(conf=conf)
    sq = SQLContext(sc)
    sq = HiveContext(sc)
    main(options.fileid)

    sc.stop()





