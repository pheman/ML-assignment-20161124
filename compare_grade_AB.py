from random import random
from operator import add
import subprocess, sys, os, time
import json,re
from pyspark import SparkContext, HiveContext,SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark import StorageLevel
from optparse import OptionParser
import datetime
import time

def get_pv_click(line):
    base_pv, base_click, exp_d_pv, exp_d_click, exp_e_pv, exp_e_click = 0, 0, 0, 0, 0, 0
    resp = json.loads(line['info'])
    if resp.has_key('bucket') and resp.has_key('sign') and resp.has_key('resp') and resp['sign'] == '360_79aabe15':
        bucket = resp['bucket']
        for s in resp['resp']:
            if s.has_key('source'):
                if s['source'].find('from_d') >= 0:
                    exp_d_pv += 1
                    if s.has_key('clicktime') and s['clicktime'] != '': exp_d_click += 1
                if s['source'].find('from_e') >= 0:
                    exp_e_pv += 1
                    if s.has_key('clicktime') and s['clicktime'] != '': exp_e_click += 1
                base_pv +=1
           # base_pv += 1
            if s.has_key('clicktime') and s['clicktime'] != '': base_click += 1
    else:
        bucket = "error"

    return Row(bucket=bucket, base_pv=base_pv, base_click=base_click, exp_d_pv=exp_d_pv, exp_d_click=exp_d_click,
               exp_e_pv=exp_e_pv, exp_e_click=exp_e_click)


def parse_profile(line):
#    try:
        lsp = line.split('\t')
        uid_bucket = lsp[0].split('_')
        uid = uid_bucket[0]
        bucket = uid_bucket[-1]
        profile = lsp[2]
        return (uid,bucket,profile)
#    except:
#        return ('','','')
