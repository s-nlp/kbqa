#!/bin/bash

python GAP_COLING2022/cli_gap.py \
         --do_predict \
         --output_dir GAP_COLING2022/auxiliary_file/out_mintaka_xl_train\
         --train_file data/mintaka_xl_filtered/train \
         --predict_file data/mintaka_xl_filtered/train \
         --tokenizer_path facebook/bart-base \
         --dataset webnlg \
         --entity_entity \
         --entity_relation \
         --type_encoding \
         --max_node_length 50 \
         --train_batch_size 16 \
         --predict_batch_size 16 \
         --max_input_length 256 \
         --max_output_length 128 \
         --append_another_bos \
         --learning_rate 2e-5 \
         --num_train_epochs 40 \
         --warmup_steps 1600 \
         --eval_period 500 \
         --num_beams 5 \
         --prefix train_mintaka_xl_
