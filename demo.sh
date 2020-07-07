#!/bin/bash

### Train

# FP32
python train.py --epochs 25 \
                --cfg cfg/yolov3_coco_person.cfg \
                --data data/coco_person.data \
                --weights weights/yolov3.pt \
                --name COCO_PERSON_FP32_NAG_EVVD_1 \
                --multi-scale \
                --device 0 \
                --hyps hyps/fp32_nag_evd_1.py \
                --batch-size 8

# HFP8
python train.py --epochs 25 \
                --cfg cfg/yolov3_coco_person.cfg \
                --data data/coco_person.data \
                --weights weights/yolov3.pt \
                --name COCO_PERSON_HFP8_26_EVVD_1 \
                --multi-scale \
                --device 0 \
                --hyps hyps/hfp8_26_evd_1.py \
                --batch-size 8 \
                --quant

###

### Test

# FP32
python test.py --cfg cfg/yolov3_coco_person.cfg \
               --data data/coco_person.data \
               --weights weights/best_COCO_PERSON_FP32_NAG.pt \
               --device 0 --hyps hyps/fp32_nag.py \
               --batch-size 8

# HFP8

python test.py --cfg cfg/yolov3_coco_person.cfg \
               --data data/coco_person.data \
               --weights weights/COCO_PERSON_HFP8_26_EVVD_1_best.pt \
               --device 0 --hyps hyps/hfp8_26_evd_1.py \
               --batch-size 8 \
               --quant

###

### Detection

python detect.py --cfg cfg/yolov3_coco_person.cfg \
                 --names data/coco_person.names \
                 --weights weights/COCO_PERSON_HFP8_26_EVVD_1_best.pt \
                 --device 0 \
                 --source videos/test_arin_dist_big.mp4 \
                 --quant \
                 --hyps hyps/hfp8_26_evd_1.py

###

### Social Distancing

# Manual
python detect.py --cfg cfg/yolov3_coco_person.cfg \
                 --names data/coco_person.names \
                 --weights weights/COCO_PERSON_HFP8_26_EVVD_1_best.pt \
                 --device 0 \
                 --source videos/test_arin_dist_big.mp4 \
                 --quant \
                 --hyps hyps/hfp8_26_evd_1.py \
                 socialdistancing --bird-scale 1 1

# Auto
python detect.py --cfg cfg/yolov3_coco_person.cfg \
                 --names data/coco_person.names \
                 --weights weights/COCO_PERSON_HFP8_26_EVVD_1_best.pt \
                 --device 0 \
                 --source videos/test_arin_dist_big.mp4 \
                 --quant \
                 --hyps hyps/hfp8_26_evd_1.py \
                 socialdistancing --camera-params calibration/camera_params/ \
                                  --bird-scale 1 1

###