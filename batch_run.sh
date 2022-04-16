#!/bin/bash
for (( counter=0; counter<50; counter++ ))
  do
  #CUDA_VISIBLE_DEVICES=1 python main.py --config config/pca/avgpool_linear_pca5_f.yml --proj_dir project/pca/sgd_l2_avg_5_f
  CUDA_VISIBLE_DEVICES=3 python main_am.py --config config/anths/conf_linear_anths_f.yml --proj_dir project/anths/sgd_10086_l2_conf_f_wd001
  #CUDA_VISIBLE_DEVICES=1 python main_am.py --config config/anths/conf_linear_anths_m.yml --proj_dir project/anths/sgd_10086_l2_conf_m
done
