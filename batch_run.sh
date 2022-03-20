#!/bin/bash
for (( counter=0; counter<50; counter++ ))
  do
  CUDA_VISIBLE_DEVICES=3 python main_am.py --config config/conf_linear_anth.yml --measure_indices "51" --proj_dir project/seed_666_adam_dropout/dropout_adam_l2_vfn_51
done
