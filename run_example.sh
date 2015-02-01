#! /bin/bash

python cnn4nlp.py   --l2  --norm_w --ebd_dm 48 --ebd_delay_epoch=0 --au=tanh 	--batch_size=10 	--dr 0.5 0.5 0.5 0.5     --fold 1 1 1     --l2_regs 1e-4 3e-5 3e-6 1e-5 1e-4  --ks 20 10 2   --conv_layer_n=3     --nkerns 5 10 18     --filter_widths 6 5 3

