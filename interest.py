#!/bin/sh
source ~/.bashrc

date_id=$(date +%Y%m%d)

day=2
while true
do
    last_profile_date=`date -d -${day}day '+%Y%m%d'`
    hadoop fs -test -e /home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/merged_profile/user_interest.${last_profile_date}/_SUCCESS
    if [ $? -eq 0 ]; then
        echo last_profile_date=${last_profile_date}
        break
    else
        echo ${last_profile_date}
        echo ${last_profile_date} does not exist
        ((day=day+1))
    fi
done

day=2
wd_log_date=`date -d -${day}day '+%Y%m%d'`
hadoop fs -test -e /home/nlp/offline/right_recom/safebox/nlp_correlation/${wd_log__date}/0000/query_coocur/_SUCCESS
if [ $? -eq 0 ]; then
    echo wd_log_date=${wd_log_date}, run get_interest
    break
else
    echo no new log, exit
    exit 0
fi

#while true
#do
#    sleep 5
#    echo looping
#done

sparksubmit  get_interest_json_dict.py  \
              --last_profile_date=${last_profile_date} \




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

def str2int(s):
    try:
        return int(s)
    except ValueError:
        return 0

def get_host_list(filename):
    f = open(filename)
    lines = f.readlines()
    hostList = []
    for line in lines:
        hostList.append( line.split('__')[-1].strip() )      
    return  set(hostList)


def get_tag_filter_list():
    f = open('kickOffTopTag_12.26.csv')
    lines = f.readlines()
    tagList = []
    for line in lines:
        tagList.append(line.strip().decode('utf-8'))
    return set(tagList)
#tagList = get_tag_filter_list()


def get_url(line):
    line_split = line.split('\t')
    uid  = line_split[0].split('\x01')[0][1:]
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
    
    uid_host_url_pair = [(uid, host_url_pair[0], host_url_pair[1]) for host_url_pair in host_url]
    
    #return Row(uid = uid, date=date, numItems=numItems, http=http, query=query, other=other, urlList=validList)
    return uid_host_url_pair

def parse_tag(line):
    line = line.split('\t')
    try:
        url = line[0]
        host = url.split('http://')[1].split('/')[0]
        return (url,host, line[1], line[2])
    except:
        return (line[0], '','', '')

def itt_weight(row):
    uid = row['uid']
    taglist = row['interest']
    tagDict = {}
    for tagGroup in taglist:
        for tag in tagGroup.split('|'):
            if tag=='': continue
            if tag in tagDict:   tagDict[tag] += 1
            else:                tagDict[tag]  = 1

    return (uid, json.dumps( tagDict, ensure_ascii=False))

def parse_ui_dict(line):
    lsp = line.split('\t')
    uid = lsp[0]#[1:]
    tagDict = lsp[1]
    return (uid, tagDict)

def mergeDict(lline):
    l = [json.loads(x) for x in lline]
    if len(l) == 1:
        return json.dumps( l[0], ensure_ascii=False)
    else:
        for d in l[1:]:
            for key in d:
                try: 
                    l[0][key] += d[key]
                except: 
                    l[0][key]  = d[key]
        return json.dumps( l[0], ensure_ascii=False)

def main( last_profile_date):
    ifall = ''
    base_dir = 'hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000'
    async_dir = base_dir + '/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/async'
    utt_path  = async_dir + '/url_title_tag/utt*/part-{0}*'.format(ifall)
    utt_path  = async_dir + '/utt_all/part-{0}*'.format(ifall)

    tagList = get_tag_filter_list()
    tagList_bd = sc.broadcast(tagList)
    zxhostList = get_host_list('zsHostList_12.26.csv')
    zxhost_bd = sc.broadcast(zxhostList)
    utt  = sc.textFile(utt_path).map(parse_tag).toDF(['url','host','title','tag'])
    utt = utt.filter(utt.tag != '').dropDuplicates().select('url','tag').coalesce(100)
#    utt = utt.filter(utt.tag != '').filter(~utt.tag.isin(tagList_bd.value)).dropDuplicates().select('url','tag').persist()
#    utt = utt.filter(utt.tag != '').filter(~utt.tag.isin(tagList_bd.value)).filter( utt.host.isin(zxhost_bd.value)).dropDuplicates().select('url','tag').persist()

    hostList = get_host_list('hostFrom')
    hostList_bd = sc.broadcast(hostList)

    date = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y%m%d")
    log_path = base_dir + '/home/nlp/offline/right_recom/safebox/nlp_correlation/{date}/0000/query_coocur/part-{ifall}*'\
                          .format(date=date, ifall=ifall)
    urllog = sc.textFile(log_path)
    urllog = urllog.flatMap(get_url).toDF(['uid','host','url'])
    urllog = urllog.filter(urllog.host.isin(hostList_bd.value))#.coalesce(2000)
    urllog = urllog.join(utt, 'url', 'left')
    urllog = urllog.filter(F.isnull('tag') == False).filter(urllog.tag != '').coalesce(1000)
            
    uid_interest_list = urllog.groupby('uid').agg( F.collect_list(urllog.tag).alias('interest'))
    uid_interest = uid_interest_list.map( itt_weight).toDF(['uid','interest'])
    
    ui_history_path = base_dir + '/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/merged_profile/user_interest.{0}/part-{1}*'.format(last_profile_date, ifall)
    ui_history = sc.textFile( ui_history_path ).map(parse_ui_dict).toDF(['uid','interest'])

    uiAll = ui_history.unionAll( uid_interest ).repartition(8000)
    #uiAll = uiAll.groupByKey().mapValues(mergeDict).map(lambda x : '\t'.join([x[0], x[1]]))
    #squdf = F.udf( mergeDict)
    uiAll = uiAll.groupby('uid').agg(F.collect_list('interest').alias('interest')).coalesce(8000)\
                                .map(lambda x : '\t'.join([x['uid'], mergeDict(x['interest'])]))
                                
                               # .select('uid',squdf('interest').alias('interest'))\
                               # .map(lambda x : '\t'.join([x['uid'], x['interest']]))
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")
    uid_interest_path = base_dir + '/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/merged_profile/user_interest.{0}'.format(yesterday)
   # uid_interest_path = base_dir + '/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/merged_profile/user_interest_debug.{0}'.format(yesterday)
    os.system('hadoop fs -rmr {sdir}'.format( sdir=uid_interest_path) )
    uiAll.coalesce(8000).saveAsTextFile( uid_interest_path, "org.apache.hadoop.io.compress.GzipCodec")


if __name__ == "__main__":
    usage = "usage: %prog [options] arg"
    
    parser = OptionParser(usage = usage)

    parser.add_option("--last_profile_date", dest='last_profile_date')
    (options, args) = parser.parse_args()

    conf = SparkConf().setAppName("zhangruibin_incremental_tag") \
                      .set("spark.driver.memory", "100g") \
                      .set("spark.executor.memory", "6g") \
                      .set("spark.storage.memoryFraction", 0.5) \
                      .set("spark.shuffle.memoryFraction", 0.4) \
                      .set("spark.network.timeout", 1200) \
                      .set("spark.executor.instances", 1000) \
                      .set("spark.executor.cores", 2) \
                      .set("spark.default.parallelism", 8000) \
                      .set("spark.yarn.priority", "VERY_HIGH") \
                      .set("spark.shuffle.service.enabled", False) \
                      .set("spark.yarn.queue", "root.hdp-reader")\
                      .set("spark.scheduler.listenerbus.eventqueue.size", 20000)\
                      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sc = SparkContext(conf=conf)
    sq = SQLContext(sc)
    sq = HiveContext(sc)
    main(options.last_profile_date)

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

def str2int(s):
    try:
        return int(s)
    except ValueError:
        return 0

def get_host_list(filename):
    f = open(filename)
    lines = f.readlines()
    hostList = []
    for line in lines:
        hostList.append( line.split('__')[-1].strip() )      
    return  set(hostList)


def get_tag_filter_list():
    f = open('kickOffTopTag_12.26.csv')
    lines = f.readlines()
    tagList = []
    for line in lines:
        tagList.append(line.strip().decode('utf-8'))
    return set(tagList)
#tagList = get_tag_filter_list()


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
    
    uid_host_url_pair = [(uid, host_url_pair[0], host_url_pair[1]) for host_url_pair in host_url]
    
    #return Row(uid = uid, date=date, numItems=numItems, http=http, query=query, other=other, urlList=validList)
    return uid_host_url_pair

def parse_tag(line):
    line = line.split('\t')
    try:
        url = line[0]
        host = url.split('http://')[1].split('/')[0]
        return (url,host, line[1], line[2])
    except:
        return (line[0], '','', '')

def itt_weight(row):
    uid = row['uid']
    taglist = row['interest']
    tagDict = {}
    for tagGroup in taglist:
        for tag in tagGroup.split('|'):
            if tag=='': continue
            if tag in tagDict:   tagDict[tag] += 1
            else:                tagDict[tag]  = 1

    return '\t'.join([uid, json.dumps(tagDict, ensure_ascii=False)])


def main():
    base_dir = 'hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000'
    async_dir = base_dir + '/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/async'
    utt_path  = async_dir + '/url_title_tag/utt*/part-*'

    tagList = get_tag_filter_list()
    tagList_bd = sc.broadcast(tagList)
    zxhostList = get_host_list('zsHostList_12.26.csv')
    zxhost_bd = sc.broadcast(zxhostList)
    utt  = sc.textFile(utt_path).map(parse_tag).toDF(['url','host','title','tag'])
    utt = utt.filter(utt.tag != '').dropDuplicates().coalesce(100).select('url','tag').persist()
#    utt = utt.filter(utt.tag != '').filter(~utt.tag.isin(tagList_bd.value)).dropDuplicates().select('url','tag').persist()
#    utt = utt.filter(utt.tag != '').filter(~utt.tag.isin(tagList_bd.value)).filter( utt.host.isin(zxhost_bd.value)).dropDuplicates().select('url','tag').persist()

    hostList = get_host_list('hostFrom')
    hostList_bd = sc.broadcast(hostList)

    for month in [ '07','08','09','10','11','12' ]:
        log_path = base_dir + '/home/nlp/offline/right_recom/safebox/nlp_correlation/2017{mon}*/0000/query_coocur/part-*'.format(mon=month)
        urllog = sc.textFile(log_path)
        urllog = urllog.flatMap(get_url).toDF(['uid','host','url'])
        urllog = urllog.filter(urllog.host.isin(hostList_bd.value)).coalesce(2000)
        urllog = urllog.join(utt, 'url', 'left')
        urllog = urllog.filter(F.isnull('tag') == False).filter(urllog.tag != '')
        try:
            urllogAll = urllogAll.unionAll(urllog)
        except:
            urllogAll = urllog
            
    uid_interest_list = urllogAll.coalesce(2000).groupby('uid').agg( F.collect_list(urllogAll.tag).alias('interest'))
    uid_interest = uid_interest_list.map( itt_weight)
    
    uid_interest_path = async_dir + '/user_interest/uid_all_interest_json_dict_01_02'
    os.system('hadoop fs -rmr {sdir}'.format( sdir=uid_interest_path) )
    uid_interest.saveAsTextFile( uid_interest_path, "org.apache.hadoop.io.compress.GzipCodec")


if __name__ == "__main__":

    conf = SparkConf().setAppName("zhangruibin_join_tag") \
                      .set("spark.driver.memory", "500g") \
                      .set("spark.executor.memory", "7g") \
                      .set("spark.storage.memoryFraction", 0.5) \
                      .set("spark.shuffle.memoryFraction", 0.4) \
                      .set("spark.network.timeout", 1200) \
                      .set("spark.executor.instances", 1000) \
                      .set("spark.executor.cores", 1) \
                      .set("spark.default.parallelism", 2000) \
                      .set("spark.yarn.priority", "VERY_HIGH") \
                      .set("spark.shuffle.service.enabled", True) \
                      .set("spark.yarn.queue", "root.hdp-reader")
    sc = SparkContext(conf=conf)
    sq = SQLContext(sc)
    sq = HiveContext(sc)
    main()

    sc.stop()





