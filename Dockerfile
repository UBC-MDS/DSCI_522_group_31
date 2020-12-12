# Dockerfile for DSCI-522-Group-31 project
# Author: Mai Le
# 12/11/2020
# build: docker build --tag dsci522 .
# run: docker run -it --rm -v YOUR_LOCAL_PROJECT_PATH:/home/online_shopping_predictor dsci522 make -C /home/online_shopping_predictor all
# OR
# docker run --rm -v YOUR_LOCAL_PROJECT_PATH:/home/online_shopping_predictor lephanthuymai/dsci_522_group_31:latest make -C /home/online_shopping_predictor all

# use ubuntu base to use bash file
FROM debian:stable
RUN apt-get update
RUN apt-get install -y nodejs

# use jupyter/scipy-notebook as the base image
FROM jupyter/scipy-notebook

# need root access to run apt-get update, apt-get install
ARG SSL_KEYSTORE_PASSWORD
USER root

# needed for installing Chrome
RUN apt-get update && apt-get install -y gnupg2

# install docopt python package
RUN conda install -y -c anaconda \ 
    docopt \
    requests \
    imbalanced-learn\
    altair\
    altair_saver\
    arrow-cpp\
    pyarrow\
    r-arrow\
    r-rmarkdown\
    r-tidyverse\
    pandoc\
    pandocfilters\
    ipykernel\
    pip>=20\
    python-chromedriver-binary\
    feather-format

# pip install required libraries
RUN pip install psutil>=5.7.2\
    git+git://github.com/mgelbart/plot-classifier.git\
    chromedriver-binary-auto


# install Chrome to use with altair save
RUN \
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*