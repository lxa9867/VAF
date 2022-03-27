#!/bin/bash
for (( counter=0; counter<10; counter++ ))
  do
  CUDA_VISIBLE_DEVICES=2 python main.py --config config/raw_vertex/avgpool_linear_vtx_m.yml --proj_dir project/raw_vertex/sgd_l2_avg_m
done
