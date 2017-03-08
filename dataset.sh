#!/bin/bash

DATA_ROOT=/mnt/dataset/EMNIST/

cat $DATA_ROOT/emnist-balanced-test-labels-idx1-ubyte | od -An -v -tu1 -j8 -w1 | tr -d ' ' > $DATA_ROOT/emnist-balanced-test-labels-idx1-txt
cat $DATA_ROOT/emnist-balanced-test-images-idx3-ubyte | od -An -v -tu1 -j16 -w784 | sed 's/^ *//' | tr -s ' ' > $DATA_ROOT/emnist-balanced-test-images-idx3-txt

cat $DATA_ROOT/emnist-balanced-train-labels-idx1-ubyte | od -An -v -tu1 -j8 -w1 | tr -d ' ' > $DATA_ROOT/emnist-balanced-train-labels-idx1-txt
cat $DATA_ROOT/emnist-balanced-train-images-idx3-ubyte | od -An -v -tu1 -j16 -w784 | sed 's/^ *//' | tr -s ' ' > $DATA_ROOT/emnist-balanced-train-images-idx3-txt
