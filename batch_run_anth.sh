#!/bin/bash
gpu=5
measid=13
# CUDA_VISIBLE_DEVICES=$gpu python main_am.py --config config/anth/avgpool_linear_anth_m.yml --measure_indices $measid --proj_dir project/anth/sgd_666_l2_avg_m/measid_$measid
for (( counter=0; counter<30; counter++ ))
  do
  CUDA_VISIBLE_DEVICES=$gpu python main_am.py --config config/anth/avgpool_linear_anth_m.yml --measure_indices $measid --proj_dir project/anth/sgd_666_l2_avg_m/measid_$measid
done
