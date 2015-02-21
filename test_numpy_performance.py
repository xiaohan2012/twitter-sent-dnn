
from dcnn import DCNN

from util import load_data
from param_util import load_dcnn_model_params

params = load_dcnn_model_params("models/filter_widths=8,6,,batch_size=10,,ks=20,8,,fold=1,1,,conv_layer_n=2,,ebd_dm=48,,l2_regs=1e-06,1e-06,1e-06,0.0001,,dr=0.5,0.5,,nkerns=7,12.pkl")

model = DCNN(params)


datasets = load_data("data/twitter.pkl")

dev_set_x, dev_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

dev_set_x, dev_set_y = dev_set_x.get_value(), dev_set_y.get_value()
test_set_x, test_set_y = test_set_x.get_value(), test_set_y.get_value()

print "dev error:", model._errors(dev_set_x, dev_set_y)
print "test error:", model._errors(test_set_x, test_set_y)
