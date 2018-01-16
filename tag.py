#!/bin/sh

source ~/.bashrc

fileid=1345
base_dir=hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000
async_dir=${base_dir}"/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/async"
tryNum=0
while true
#for i in {0..4}
do
  hadoop fs -test -e ${async_dir}"/url_title_tag/utt_"${fileid}"/_SUCCESS"
  if [ $? -eq 0 ]; then
    echo file ${fileid} already finished
   ((fileid=fileid+1))
  else
    hadoop fs -test -e ${async_dir}"/url_title/url_title_"${fileid}"/_SUCCESS"
    if [ $? -eq 0 ]; then
         echo sparksubmit get_tag.py --fileid ${fileid}
         sparksubmit get_tag.py --fileid ${fileid}
         echo hadoop fs -mv ${async_dir}"/url_title_tag/tmp_"${fileid} \
                            ${async_dir}"/url_title_tag/utt_"${fileid}
         hadoop fs -mv ${async_dir}"/url_title_tag/tmp_"${fileid} \
                       ${async_dir}"/url_title_tag/utt_"${fileid}
    #     hadoop fs -rmr ${async_dir}"/url_title/url_title_"${fileid}
         echo file ${fileid} finished
         ((fileid=fileid+1))
    else
         hadoop fs -test -e ${async_dir}"/url_title/url_title_"${fileid}"/_temporary"
         if [ $? -eq 0 ]; then
             if [ tryNum -lt 10 ]; then
                 ((tryNum=tryNum+1))
                  echo try the ${tryNum} th time
             else
                 echo url_title fileid=${fileid} failed, pass this file, tried ${tryNum} times
                 ((fileid=fileid+1))
                 tryNum=0
             fi
         else
             echo waiting for url_title files ${fileid}
             sleep 60
         fi 
    fi
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
import base64,json
import hashlib
import urllib2
import urllib
import re

def parse2(line):
    line = line.split('\t')
    try:
        return (line[0], line[1])
    except:
        return (line[0], '')

def parse3(line):
    linesp = line.split('\t')
    try:
        return (linesp[0], linesp[1], linesp[2])
    except:
        return (linesp[0], '', '')


def call_tag(line):
    title = line[1]
    if len(title) < 5:
        titlequote = ' '.join([ title, title, title ])
    else:
        titlequote = title

    quote_title = urllib.quote(titlequote.encode('utf-8'))
    url = line[0]
    tagurl = 'http://ctr09.adsys.zzzc.qihoo.net:19528/mod_content/MyProcess?title={0}&url=zhangruibin.com'.format(quote_title)
    try:
        req = urllib2.Request(tagurl)
        res = urllib2.urlopen(req, timeout=100).read()
        res = json.loads(res)
#        if 'index_st' in res and res['index_st'] != '':
#            tag = res['index_st']#.encode('raw_unicode_escape').decode('utf-8')
#        elif 'index_all' in res:
#            index_list = res['index_all'].split('|')
#            tag = '|'.join([i for i in index_list if re.search('k(.*)_1', i)])   
#        else:
#            tag = ''
        tag = res['index_all']
    except:
        tag = ''
    
    return ( url, title, tag )


def main(fileid):
    base_dir = 'hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000'
    async_dir = base_dir + '/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/async'
    url_title_path    = async_dir + '/url_title/url_title_{0}/part-*'.format(fileid)
    utt_path = base_dir + '/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/async/url_title_tag/utt_*/part-*'
    url_title_tag_tmp_path = async_dir + '/url_title_tag/tmp_{0}'.format(fileid)
    
    url_title  = sc.textFile(url_title_path).map(parse2).toDF(['url','title'])
    try:
        utt = sc.textFile(utt_path).map(parse3).toDF(['url','title','tag'])
        utt = utt.filter(utt.tag != '').dropDuplicates().coalesce(100)

        url_title_join = url_title.join(utt, 'url', 'left')
        url_title_join_new = url_title_join.filter(F.isnull('tag'))
        url_title_new = url_title_join_new.map(lambda x : (x['url'], x['title']))

        ut_num  = url_title.count()
        utj_num = url_title_join.count()
        new_num = url_title_join_new.count()
        print 'ut, join and new num is ', ut_num,utj_num,new_num
    except:
        url_title_new = url_title.map(lambda x : (x['url'], x['title']))
  
    url_tag = url_title_new.repartition(800).map(call_tag).toDF( ['uid', 'title', 'tag'])
#    url_tag = url_tag.filter(url_tag.tag != '').dropDuplicates()
    url_tag = url_tag.map( lambda x: '\t'.join( [ x.uid, x.title, x.tag ] ) )
    os.system('hadoop fs -rmr {sdir}'.format( sdir=url_title_tag_tmp_path ) )
    print 'output dir', url_title_tag_tmp_path
    url_tag_num = url_tag.count()
    print 'tag num', url_tag_num
    url_tag.saveAsTextFile(url_title_tag_tmp_path, "org.apache.hadoop.io.compress.GzipCodec")


if __name__ == "__main__":
    usage = "usage: %prog [options] arg" 
    parser = OptionParser(usage = usage)
    parser.add_option("--fileid", dest='fileid',  help='fileid')
    (options, args) = parser.parse_args()
    conf = SparkConf().setAppName("zhangruibin_get_tag") \
                      .set("spark.driver.memory", "100g") \
                      .set("spark.driver.maxResultSize", "2g")\
                      .set("spark.executor.memory", "2g") \
                      .set("spark.storage.memoryFraction", 0.4) \
                      .set("spark.shuffle.memoryFraction", 0.5) \
                      .set("spark.network.timeout", 1200) \
                      .set("spark.executor.instances", 300) \
                      .set("spark.executor.cores", 2) \
                      .set("spark.default.parallelism", 4000) \
                      .set("spark.yarn.priority", "VERY_HIGH") \
                      .set("spark.shuffle.service.enabled", True) \
                      .set("spark.yarn.queue", 'root.hdp-reader') # root.hdp-reader
    sc = SparkContext(conf=conf)
    sq = SQLContext(sc)
    sq = HiveContext(sc)
    main(options.fileid)

    sc.stop()



