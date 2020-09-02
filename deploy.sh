#!/bin/bash
if [ ! -n "$1" ] ;then
  echo "please input SIT or PRD"
else
  echo "the word you input is $1"
fi

today=` date "+%Y%m%d"`
abbrev=`git log|head -1|cut -b 8-14`
tag=$today-$abbrev
docker build -t registry-vpc.cn-beijing.aliyuncs.com/visva/nlp-segment:$tag .
if test `docker ps |grep nlp-segment|wc -l` == 1
 then
  echo 'container exists'
  docker rm -f nlp-segment
else
  echo 'container not  exists'
fi
#docker run -e "profile=$1"  -d  --name nlp-segment -p 10020:10020 --restart=always registry-vpc.cn-beijing.aliyuncs.com/visva/nlp-segment:$tag

if [ -n "$2" ] && [ "$2" = "push" ] ;then
  echo $2
  echo "registry-vpc.cn-beijing.aliyuncs.com/visva/nlp-segment:$tag to docker hub------"
  docker push registry-vpc.cn-beijing.aliyuncs.com/visva/nlp-segment:$tag
fi
