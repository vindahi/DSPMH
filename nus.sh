#!/bin/bash
set -e
METHOD='DSPMH'
bits=(16 32 64 128)

for i in ${bits[*]}; do
  echo "**********Start ${METHOD} algorithm**********"
  CUDA_VISIBLE_DEVICES=0 python train.py --nbit $i \
                                                     --dataset nuswide \
                                                     --epochs 140 \
                                                     --dropout 0.5 \

  echo "**********End ${METHOD} algorithm**********"
done
