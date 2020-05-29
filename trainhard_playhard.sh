#!/bin/bash
set -x

dev=$1


ep=2

hypss=(
    # "hyps/hfp8_20.py"
    # "hyps/hfp8_21.py"
    # "hyps/hfp8_22.py"
    "hyps/hfp8_23.py"
    "hyps/hfp8_24.py"
)

names=(
    # "SVHN_HFP8_20"
    # "SVHN_HFP8_21"
    # "SVHN_HFP8_22"
    "SVHN_HFP8_23"
    "SVHN_HFP8_24"
)

for i in ${!hypss[@]}
do
    echo ${hyps[$i]}
    python train.py --epochs $ep \
                    --cfg /data/SVHN/yolov3-svhn-9anchors-192-multiscale-1.cfg \
                    --data /data/SVHN/train_test.data \
                    --img-size 192 \
                    --weights weights/yolov3.pt \
                    --name ${names[$i]} \
                    --multi-scale \
                    --device $dev \
                    --hyps ${hypss[$i]} \
                    --quant;  # IMPRORTANT!
done

# bash clean_tb_logs.sh '*FP*'