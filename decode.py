#!/bin/sh

source ~/.bashrc

fileid=1345
base_dir=hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000
async_dir=${base_dir}"/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/async"

while true
#for i in {0..4}
do
  hadoop fs -test -e ${async_dir}"/url_title/url_title_"${fileid}"/_SUCCESS"
  if [ $? -eq 0 ]; then
    echo fileid=${fileid} already finished, pass
    ((fileid=fileid+1))
  else
    hadoop fs -test -e ${async_dir}"/url_save/url_list_"${fileid}"/_SUCCESS"
    if [ $? -eq 0 ]; then
         cd /home/nlp/hbase-mr-tools-new/hbase-mr-tools/bin
         sh select-by-keys.sh  zhangruibin_select_hbase \
                               ${async_dir}"/url_save/url_list_"${fileid} \
                               ${async_dir}"/hbase_save/hbase_list_"${fileid} \
                               webpage true  'text:text'     'text:text'
        #hadoop fs -rmr ${async_dir}"/url_save/url_list_"${fileid}

         cd -
         sparksubmit decode_title.py --fileid ${fileid}
         echo sparksubmit decode_title.py --fileid ${fileid}
        #hadoop fs -rmr ${async_dir}"/hbase_save/hbase_list_"${fileid}
         echo file ${fileid} finished
         ((fileid=fileid+1))
    else
         sleep 10
         echo waiting for url list ${fileid}
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
import subprocess, sys, os, time, re
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

def parse3(line):
    line = line.split('\t')
    try:
        return (line[0], line[1], line[2])
    except:
        return (line[0], '' , '' )

def dec(info):
    info0 = info
    info    = base64.b64decode(info).split(',')
    for content in info:
        if '"realtitle":' in content:
            return content.split(':')[-1][0:-1]
    if len(info) > 2 and '"title"' in info[2]:
        try:
            content = info[2]    
            content = content.split('"txt":')
            content[0] = '{' + content[0]
            content[1] = '"' + content[1]
            content[1] = content[1][0:-2] + '"}]}'
            content = '"txt":'.join(content)
            fulltitle = json.loads(content)['title'][0]['txt']
            try:
                realtitle = re.match('([^_|-]+[_|-]{1}){1,1}?',fulltitle).group()[0:-1]
                return realtitle.strip("'")
            except:
                return fulltitle.strip("'")
        except:
            try:
                content = info[2].decode('utf-8')
                content = content.split('"txt":')
                content[0] = '{' + content[0]
                content[1] = '"' + content[1]
                content[1] = content[1][0:-2] + '"}]}'
                content = '"txt":'.join(content)
                try:
                    fulltitle = json.loads(content)['title'][0]['txt']
                    try:
                        realtitle = re.match('([^_|-]+[_|-]{1}){1,1}?',fulltitle).group()[0:-1]
                        return realtitle.strip("'")
                    except:
                        return fulltitle.strip("'")
                except:
                    fulltitle = content.split(':')[-1][0:-5]
                    try:
                        realtitle = re.match('([^_|-]+[_|-]{1}){1,1}?',fulltitle).group()[0:-1]
                        return realtitle.strip("'")
                    except:
                        return fulltitle.strip("'")
            except:
                return ''
    elif len(info) > 0 and '"title"' in info[0]:
        sget = info[0]
        fulltitle = re.search('(<title:[^>]+>)+?',sget).group()[7:-1]
        realtitle = re.match('([^_|-]+[_|-]{1}){1,1}?',fulltitle).group()[0:-1]
        return fulltitle.strip("'")
    else:
        return ''

       
def parse_hbase(line):
    line = line.split('\t')
    url  = line[0]
    title = dec(line[1])
    try:
        title = title.split(':')[-1]
        res = '\t'.join([ url, title] )
        return res
    except:
        try:
            title = title.decode('utf-8')
            title = title.split(':')[-1]
            res = '\t'.join([ url, title] )
            return res
        except:
            res = '\t'.join([ url, ''] )
            return res


def main(fileid):
    base_dir = 'hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000'
    async_dir = base_dir + '/home/hdp-reader/proj/hdp-reader-up/personal/zhangruibin/async'
    url_title_path   =async_dir + '/url_title/url_title_{0}'.format(fileid)
    title_hbase_path  = async_dir + '/hbase_save/hbase_list_{0}'.format(fileid)
    
    title_hbase  = sc.textFile(title_hbase_path)
    
    url_title = title_hbase.map(parse_hbase)
    os.system('hadoop fs -rmr {sdir}'.format( sdir=url_title_path ) )
    url_title.saveAsTextFile(url_title_path, "org.apache.hadoop.io.compress.GzipCodec")

if __name__ == "__main__":
    usage = "usage: %prog [options] arg" 
    parser = OptionParser(usage = usage)
    parser.add_option("--fileid", dest='fileid',  help='fileid')
    (options, args) = parser.parse_args()
    conf = SparkConf().setAppName("zhangruibin_decode") \
                      .set("spark.driver.memory", "100g") \
                      .set("spark.driver.maxResultSize", "2g")\
                      .set("spark.executor.memory", "2g") \
                      .set("spark.storage.memoryFraction", 0.4) \
                      .set("spark.shuffle.memoryFraction", 0.5) \
                      .set("spark.network.timeout", 1200) \
                      .set("spark.executor.instances", 500) \
                      .set("spark.executor.cores", 2) \
                      .set("spark.default.parallelism", 4000) \
                      .set("spark.yarn.priority", "VERY_HIGH") \
                      .set("spark.shuffle.service.enabled", True) \
                      .set("spark.yarn.queue", 'root.nlp') # root.hdp-reader
    sc = SparkContext(conf=conf)
    sq = SQLContext(sc)
    sq = HiveContext(sc)
    main(options.fileid)

    sc.stop()




