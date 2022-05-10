#!/bin/bash
for (( counter=0; counter<50; counter++ ))
  do
  CUDA_VISIBLE_DEVICES=0 python main_am.py --config config/anths/conf_linear_anths_f.yml --proj_dir project/anths/sgd_666_l2_conf_f_half
  #CUDA_VISIBLE_DEVICES=1 python main_am.py --config config/anths/conf_linear_anths_f.yml --proj_dir project/anths/sgd_2333_l2_conf_f
  #CUDA_VISIBLE_DEVICES=2 python main_am.py --config config/anths/conf_linear_anths_f.yml --proj_dir project/anths/sgd_2333_l2_conf_f
  #CUDA_VISIBLE_DEVICES=3 python main_am.py --config config/anths/conf_linear_anths_m.yml --proj_dir project/anths/sgd_2333_l2_conf_m
  #CUDA_VISIBLE_DEVICES=4 python main_am.py --config config/anths/conf_linear_anths_m.yml --proj_dir project/anths/sgd_2333_l2_conf_m
  #CUDA_VISIBLE_DEVICES=5 python main_am.py --config config/anths/conf_linear_anths_m.yml --proj_dir project/anths/sgd_2333_l2_conf_m
done
