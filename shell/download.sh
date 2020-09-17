#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
export PYTHONUNBUFFERED=1
export PATH=/opt/miniconda3/envs/venv-py3/bin:$PATH

# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return

if [ ! -d $baseDir/../data ]; then
    mkdir $baseDir/../data
fi

cd $baseDir/../data
wget https://static-public.chatopera.com/ml/triplenet_data/data_for_triplenet.tar.gz -O data.tgz
tar xzf data.tgz
mv data/* .
rm -rf data data.tgz

wget https://static-public.chatopera.com/ml/triplenet_data/DoubanConversaionCorpus.zip -O douban_corpus.zip
unzip douban_corpus.zip
mv DoubanConversaionCorpus/* douban
