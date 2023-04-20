#!/usr/bin/env bash
EMBEDDING_DIR=./embedding

if [ ! -f  ./embedding/glove.840B.300d.txt ]; then
  echo "Please download the GloVe embedding first"
  exit 0
fi

python train.py \
    --batch_size 50 --epochs 50