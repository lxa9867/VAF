#!/bin/bash
for (( counter=0; counter<50; counter++ ))
  do
  CUDA_VISIBLE_DEVICES=0 python main_am.py --config config/conf_linear_anth.yml --measure_indices "12" --proj_dir project/flame_sgd_l2_721_vfn_12
