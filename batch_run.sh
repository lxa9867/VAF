#!/bin/bash
for (( counter=16*5; counter<16*6; counter++ ))
  do
  CUDA_VISIBLE_DEVICES=2 python main_am.py --config config/conf_linear_anth.yml --measure_indices "$counter"
done

