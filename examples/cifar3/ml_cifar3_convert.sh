#!bash

rm -rf cifar3_*

python ml_cifar3_convert.py

CAFFE_ROOT=/Users/riya/caffe
TOOLS=build/tools
EXAMPLE=examples/cifar3

$CAFFE_ROOT/$TOOLS/compute_image_mean $CAFFE_ROOT/$EXAMPLE/cifar3_train_lmdb \
  $CAFFE_ROOT/$EXAMPLE/mean.binaryproto
