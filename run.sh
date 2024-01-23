#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python ./model/InstDiffEdit.py \
--input_path ./dataset_txt/Editing-Mask_list.txt \
--dataset_name  Editing-Mask \
--dataset_path  ./dataset/Editing-Mask/image/ \
--output_path  ./result/ \
--allresult_path total \
--strength 0.5 \
--threshold 0.2 \
--batch_size 3 \
--or_save  True \
--mask_save True \
--seed 4623523532252