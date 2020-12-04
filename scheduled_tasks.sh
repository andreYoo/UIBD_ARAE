#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python train_gru_128.py
CUDA_VISIBLE_DEVICES=1 python train_gru_64.py
CUDA_VISIBLE_DEVICES=1 python train_rnn_64.py

