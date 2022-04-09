#!/bin/bash
gpu=0
for (( measid=12; measid<13; measid++ ))
  do
  for (( counter=0; counter<5; counter++ ))
    do
    CUDA_VISIBLE_DEVICES=$gpu python main_am.py --config config/anth/conf_linear_anth_f.yml --measure_indices $measid --proj_dir project/anth/sgd_666_l2_conf_f/measid_$measid
  done
done
