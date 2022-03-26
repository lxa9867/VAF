#!/bin/bash
for (( counter=0; counter<5; counter++ ))
  do
  # CUDA_VISIBLE_DEVICES=4 python main_am.py --config config/conf_linear_anth.yml --measure_indices "51" --proj_dir project/sgd_666_l2_vfn/measid_51
  #CUDA_VISIBLE_DEVICES=3 python main_am.py --config config/anth_exp/conf_linear_anth_wd0005_1618.yml --proj_dir project/sgd_666_l2_vfn_measid12_wd0005_1618/
  CUDA_VISIBLE_DEVICES=0 python main_am.py --config config/anth_exp/avgpool_linear_anth_wd0005.yml --proj_dir project/sgd_666_l2_vfn_avg_measid12_wd0005/
done
