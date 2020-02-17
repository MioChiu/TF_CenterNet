# common
classes_file = './data/classes/voc.names'
num_classes = 20
input_image_h = 448
input_image_w = 448
down_ratio = 4
max_objs = 150

# train
train_data_file = './data/dataset/voc_train.txt'
batch_size = 16
epochs = 80
 # learning rate
lr_type="piecewise"# "exponential","piecewise","CosineAnnealing"
lr = 1e-3               # exponential
lr_decay_steps = 5000   # exponential
lr_decay_rate = 0.95    # exponential
lr_boundaries = [40000,60000]               # piecewise
lr_piecewise = [0.0001, 0.00001, 0.000001]  # piecewise
warm_up_epochs = 2  # CosineAnnealing
init_lr= 1e-4       # CosineAnnealing
end_lr = 1e-6       # CosineAnnealing
pre_train = True

# test
test_data_file = './data/dataset/voc_test.txt'
score_threshold = 0.3
use_nms = True
nms_thresh = 0.4




