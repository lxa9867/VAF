#!/bin/bash
for (( counter=0; counter<50; counter++ ))
  do
  CUDA_VISIBLE_DEVICES=5 python main_am.py --config config/conf_linear_anth.yml --measure_indices "51" --proj_dir project/sgd_666_l2_vfn/measid_51
done
