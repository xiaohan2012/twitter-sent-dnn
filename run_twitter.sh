#! /bin/bash

# run trianing/testing on twitter

python cnn4nlp.py --corpus_path=data/twitter.pkl --model_path=models/twitter.pkl --l2  --norm_w --ebd_delay_epoch=0 --au=tanh --n_epochs=3 --filter_widths 6 5 3 --batch_size 10 --ks 20 10 5 --fold 0 0 0 --conv_layer_n 3 --ebd_dm 48 --nkerns 3 4 2 --dr 0.5 0.5 0.5 --l2_regs 1e-4 3e-5 3e-6 1e-5 1e-4 --img_prefix=twitter
