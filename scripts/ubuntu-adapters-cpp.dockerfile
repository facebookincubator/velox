ARG base

FROM ${base}

RUN apt update && \
      apt install -y \
      curl \
      libcurl4-openssl-dev \
      libkrb5-dev \
      libprotobuf-dev \
      protobuf-compiler

ADD scripts /velox/scripts/

# Minimum AWS_SDK_VERSION working for Ubuntu 22.04
ENV AWS_SDK_VERSION=1.9.379

RUN mkdir /deps
ENV DEPENDENCY_DIR=/deps

WORKDIR /velox
RUN /velox/scripts/setup-adapters.sh
