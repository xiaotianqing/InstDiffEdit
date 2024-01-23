#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python ./model/sample_use.py \
--input_path ./image/rubby.jpg \
--output_path ./  \
--prompt "a photo of a cat" \
--output_name inpainting \
--strength 0.5 \
--threshold 0.3 \
--batch_size 3 \
--mask_path None \
--or_save True \
--mask_save True \
--seed 4623523532252\