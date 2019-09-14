ori_dim = 5*4
acc_dim = 5*3
input_dim = ori_dim
output_dim = 15*4
hid_dim=512

batch_len = 10
seq_len = 200

n_layers = 2
dropout = 0.3

traindata_path = '/data/Guha/GR/Dataset/Train'
testdata_path = '/data/Guha/GR/Dataset/Test/'
validdata_path = '/data/Guha/GR/Dataset/Validation'
train_dip_path = '/data/Guha/GR/Dataset/DIP_IMU/Train'
valid_dip_path = '/data/Guha/GR/Dataset/DIP_IMU/Validation'
human36_path = '/data/Guha/GR/Dataset/H36'
use_cuda =True
