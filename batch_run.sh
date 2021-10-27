#!/bin/bash
k=3
sz=24
for (( counter=($sz)*($k); counter<($sz)*($k + 1); counter++ ))
  do
  CUDA_VISIBLE_DEVICES=$k python main_am.py --config config/conf_linear_anth.yml --measure_indices "$counter"
done

