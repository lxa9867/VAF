#!/bin/bash
for (( counter=0; counter<10; counter++ ))
  do
  # CUDA_VISIBLE_DEVICES=2 python main.py --config config/raw_vertex/avgpool_linear_vtx_m.yml --proj_dir project/raw_vertex/sgd_l2_avg_m
  CUDA_VISIBLE_DEVICES=4 python main.py --config config/pca/avgpool_linear_pca2_f.yml --proj_dir project/pca/sgd_l2_avg_2_f
done
