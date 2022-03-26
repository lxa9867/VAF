#!/bin/bash
for (( counter=0; counter<10; counter++ ))
  do
  CUDA_VISIBLE_DEVICES=3 python main.py --config config/conf_linear_f_20.yml --proj_dir project/sgd_l2_vfn_pca20_m/
done
