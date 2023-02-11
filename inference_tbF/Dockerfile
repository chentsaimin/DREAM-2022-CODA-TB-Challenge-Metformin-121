FROM ubuntu
FROM python:3.8-slim-buster

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libsndfile1  && rm -rf /var/lib/apt/lists/*

RUN pip install joblib
RUN pip install Pandas
RUN pip install onnxruntime
RUN pip install soundfile

RUN mkdir model

COPY model ./model/
COPY inference.py ./inference.py
COPY entrypoint.sh ./entrypoint.sh

ENTRYPOINT ["sh", "./entrypoint.sh"]