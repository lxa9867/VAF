#!/bin/bash
for (( counter=0; counter<50; counter++ ))
  do
  CUDA_VISIBLE_DEVICES=0 python main_am.py --config config/conf_linear_anth.yml --measure_indices "51" --proj_dir project/sgd_l2_vfn_51
done
