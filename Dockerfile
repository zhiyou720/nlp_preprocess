FROM python:3
ENV TZ "Asia/Shanghai"

RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --upgrade pip \
  && pip install  -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com Flask \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com flask-cors \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com regex \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com torch \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com numpy \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com gevent \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com gunicorn \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com ijson \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com pyparsing \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com beautifulsoup4 \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com nltk \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com emoji \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com pymongo \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com pytorch_transformers \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com sentence_transformers \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com sklearn \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com jieba \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com gensim \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com elmoformanylangs \
  && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com control-characters

ENV NLTK_DATA=/app/data/nltk
EXPOSE 10080
COPY . /src
WORKDIR /src
RUN mkdir -p /src/logs
CMD gunicorn -c gun.py service:app
