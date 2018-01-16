# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific aliases and functions


if [[ $SUDO_USER == 'zhangruibin' ]];then
      #export PYSPARK_PYTHON=python
      #export PYSPARK_DRIVER_PYTHON=python

      # added by python2.6 installer
      #export PATH="/home/nlp/zhangruibin/python2.6/bin:$PATH"

      # added by Anaconda3 installer
      #export PATH="/home/nlp/anaconda3/bin:$PATH"
      #export PATH="/home/nlp/wenliang/sbt/bin:$PATH"

      # added by Anaconda2 2.5.0 installer
      export PATH="/home/nlp/zhangruibin/anaconda2/bin:$PATH"

      # use ipython
      #export IPYTHON=1

      alias pyspark=$SPARK_HOME/bin/pyspark
      alias sparksubmit=$SPARK_HOME/bin/spark-submit
      alias ssub=$SPARK_HOME/bin/spark-submit
      alias home='cd /home/nlp/zhangruibin/'
      alias htext='hadoop fs -text '
      alias hls='hadoop fs -ls '
      alias hdu='hadoop fs -du -h '
      alias hmkdir='hadoop fs -mkdir '
      alias hcat='hadoop fs -cat '
      alias hdel='hadoop fs -rmr'
      alias hhome='/home/nlp/personal/zhangruibin/'
      export hdir='/home/nlp/personal/zhangruibin/'
      alias yarn=/usr/bin/hadoop/software/yarn/bin/yarn

      alias start_jupyter='nohup jupyter notebook > /dev/null 2>&1 &'
fi

if [[ $SUDO_USER == '' ]];then
        hdir=/home/nlp/personal/liweili/
        ldir=hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000/home/hdp-reader/data/online/mbrowser/logs/
        gdir=/home/nlp/zhaoguozhen/news_log
        xdir=/home/nlp/online/mbroswer/front_log/res.qhupdate.com/webaccess/log_res_360reader/
        fdir=/home/nlp/online/news_merger/
        alias vim='vim -u /home/liweili/liweili/.vimrc'
        alias anaconda='/da2/nlp/liweili/software/anaconda2/bin/python'
        alias hkill='/usr/bin/hadoop/software/hadoop//bin/hadoop job -jt w-jobtracker.qss.zzbc2.qihoo.net:8021 -kill '
        alias htext='hadoop fs -text '
        export PATH=/home/nlp/anaconda2/bin:$PATH
        #export LD_LIBRARY_PATH=/usr/bin/hadoop/software/jdk1.7.0/jre/lib/amd64/server/:/usr/bin/hadoop/software/spark-1.6.0-U16-bin-2.7.1/lib/native:$LD_LIBRARY_PATH
fi

if [[ $SUDO_USER == "" ]]; then
      source /home/nlp/luoxuefeng/.bashrc
      alias vim='vim -u /home/nlp/luoxuefeng/.vimrc'
      cd /home/nlp/luoxuefeng/
fi
