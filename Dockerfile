FROM nvidia/cuda:9.0-runtime
LABEL maintainer="Kazuhiro Ota <zektbach@gmail.com>"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt update -y && apt upgrade -y

RUN apt-get install -y build-essential
RUN apt-get install -y checkinstall
RUN apt-get install -y libreadline-gplv2-dev
RUN apt-get install -y libncursesw5-dev
RUN apt-get install -y libssl-dev
RUN apt-get install -y libsqlite3-dev
RUN apt-get install -y tk-dev
RUN apt-get install -y libgdbm-dev
RUN apt-get install -y libc6-dev
RUN apt-get install -y libbz2-dev
RUN apt-get install -y zlib1g-dev
RUN apt-get install -y openssl
RUN apt-get install -y libffi-dev
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-setuptools
RUN apt-get install -y wget

RUN mkdir /tmp/Python37
WORKDIR /tmp/Python37

RUN wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tar.xz
RUN tar xvf Python-3.7.3.tar.xz
WORKDIR /tmp/Python37/Python-3.7.3
RUN ./configure
RUN make altinstall

RUN pip3.7 install --upgrade pip
COPY requirements.txt .
RUN pip3.7 install -r requirements.txt

WORKDIR /workspace
CMD ["python3.7"]
