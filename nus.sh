# #!/bin/bash
# set -e
# METHOD='DSPMH'
# bits=(16 32 64 128)

# for i in ${bits[*]}; do
#   echo "**********Start ${METHOD} algorithm**********"
#   CUDA_VISIBLE_DEVICES=0 python train.py --nbit $i \
#                                                      --dataset nuswide \
#                                                      --epochs 140 \
#                                                      --dropout 0.5 \

#   echo "**********End ${METHOD} algorithm**********"
# done


#!/bin/bash
set -e
METHOD='DSPMH'
bits=(64)
rounds=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100 102 104 106 108 110 112 114 116 118 120 122 124 126 128 130 132 134 136 138 140)

for i in ${bits[*]}; do
for j in ${rounds[*]}; do
  echo "**********Start ${METHOD} algorithm**********"
  CUDA_VISIBLE_DEVICES=0 python train.py --nbit $i \
                                                     --dataset nuswide \
                                                     --epochs $j \
                                                     --dropout 0.5 \

  echo "**********End ${METHOD} algorithm**********"
done
done