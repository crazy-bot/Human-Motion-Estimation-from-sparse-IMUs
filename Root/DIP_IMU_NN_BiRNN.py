import torch
import torch.nn as nn
import torch.optim as optim
import Config as cfg
import numpy as np
from pyquaternion import Quaternion
import  itertools

class BiRNN(nn.Module):
    def __init__(self):
        super(BiRNN,self).__init__()

        self.input_dim = cfg.input_dim
        self.hid_dim = 256
        self.n_layers = cfg.n_layers
        self.dropout = cfg.dropout

        self.relu = nn.ReLU()
        self.pre_fc = nn.Linear(cfg.input_dim , 256)
        self.lstm = nn.LSTM(256, 256, cfg.n_layers, batch_first=True, dropout=cfg.dropout,bidirectional=True)
        self.post_fc = nn.Linear(256*2,cfg.output_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, X):
        # src = [ batch size, seq len, input dim]
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        input_dim = X.shape[2]

        X = X.view(-1,input_dim)
        X = self.pre_fc(X)
        X = self.relu(X)
        X = X.view(batch_size,seq_len, -1)
        lstm_out, (_, _) = self.lstm(X)

        """lstm_out : [batch size, src sent len, hid dim * n directions]
        hidden : [n layers * n directions, batch size, hid dim]
        cell : [n layers * n directions,batch size, hid dim]
        lstm_out are always from the top hidden layer """

        fc_out = self.post_fc(lstm_out)
        return fc_out

def _loss_impl(predicted, expected):
    L1 = predicted - expected
    return torch.mean((torch.norm(L1, 2, 2)))

def preparedata(path,batch_sz):
    seq_len = 300
    data_dict = dict(np.load(path, encoding='latin1'))
    oriList = data_dict['orientation']
    poseList = data_dict['smpl_pose']
    batch_ori = []
    batch_pose = []
    for i in range(len(oriList)):
        ori = oriList[i].reshape(-1,5,3,3)
        ori_quat = np.asarray([Quaternion(matrix=ori[k, j, :, :]).elements for k, j in
                               itertools.product(range(ori.shape[0]), range(5))])
        ori_quat = ori_quat.reshape(-1, 5 * 4)
        pose = poseList[i].reshape(-1,15,3,3)
        pose_quat = np.asarray([Quaternion(matrix=pose[k, j, :, :]).elements for k, j in
                                itertools.product(range(pose.shape[0]), range(15))])
        pose_quat = pose_quat.reshape(-1, 15 * 4)
        if (len(ori_quat) != seq_len):
            mask_ori = np.repeat(Quaternion(matrix=np.eye(3, 3)).elements[np.newaxis, ...], 5, axis=0)
            mask_ori = np.array([mask_ori] * seq_len).reshape(seq_len, -1)
            mask_ori[:ori_quat.shape[0], :] = ori_quat
            ori_quat = mask_ori

            mask_pose = np.repeat(Quaternion(matrix=np.eye(3, 3)).elements[np.newaxis, ...], 15, axis=0)
            mask_pose = np.array([mask_pose] * seq_len).reshape(seq_len, -1)
            mask_pose[:pose_quat.shape[0], :] = pose_quat
            pose_quat = mask_pose

        batch_ori.append(ori_quat)
        batch_pose.append(pose_quat)

    batch_ori = list(torch.split(torch.FloatTensor(batch_ori).cuda(), batch_sz))
    batch_pose = list(torch.split(torch.FloatTensor(batch_pose).cuda(), batch_sz))
    return batch_ori,batch_pose

def train(basepath):
    batch_sz = 10
    gradient_clip = 0.1
    model = BiRNN().cuda()
    modelPath = basepath
    optimizer = optim.Adam(model.parameters(), lr=.001)
    trainpath = '/data/Guha/GR/DIPIMUandOthers/DIP_IMU_and_Others/DIP_IMU_nn/imu_own_training.npz'
    validpath = '/data/Guha/GR/DIPIMUandOthers/DIP_IMU_and_Others/DIP_IMU_nn/imu_own_validation.npz'
    # trainpath = '/data/Guha/GR/dataset/DIP_IMU_nn/imu_own_training.npz'
    # validpath = '/data/Guha/GR/dataset/DIP_IMU_nn/imu_own_validation.npz'
    f = open(modelPath + 'model_details', 'w')
    f.write(' comments: dip_imu_nn training on birnn')
    f.write(str(model))
    f.write('\n')


    epoch_loss = {'train': [],'validation':[]}
    batch_ori, batch_pose = preparedata(trainpath, batch_sz)

    print('no of batches--', len(batch_ori))
    f.write('no of batches-- {} \n'.format(len(batch_ori)))
    min_valid_loss = 0.0
    for epoch in range(50):
        ############### training #############
        train_loss = []
        model.train()
        for input,target in zip(batch_ori,batch_pose):
            output = model(input)
            loss = _loss_impl(output,target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            loss.detach()
            train_loss.append(loss.item())

        train_loss = torch.mean(torch.FloatTensor(train_loss))
        print('epoch no ----------> {} training loss {} '.format(epoch, train_loss.item()))
        f.write('epoch no ----------> {} training loss {} \n'.format(epoch, train_loss.item()))
        epoch_loss['train'].append(train_loss)
        # we save the model after each epoch : epoch_{}.pth.tar
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'epoch_loss': train_loss
        }
        torch.save(state, modelPath + 'epoch_{}.pth.tar'.format(epoch + 1))
        ############### validation ###############
        data_dict = dict(np.load(validpath, encoding='latin1'))
        oriList = data_dict['orientation']
        poseList = data_dict['smpl_pose']

        model.eval()
        valid_loss = []
        for i in range(len(oriList)):
            ori = oriList[i].reshape(-1,5,3,3)
            ori_quat = np.asarray([Quaternion(matrix=ori[k, j, :, :]).elements for k, j in
                                   itertools.product(range(ori.shape[0]), range(5))])
            ori_quat = ori_quat.reshape(-1, 5 * 4)
            pose = poseList[i].reshape(-1,15,3,3)
            pose_quat = np.asarray([Quaternion(matrix=pose[k, j, :, :]).elements for k, j in
                                    itertools.product(range(pose.shape[0]), range(15))])
            pose_quat = pose_quat.reshape(-1, 15 * 4)

            input = torch.FloatTensor(ori_quat).unsqueeze(0).cuda()
            target = torch.FloatTensor(pose_quat).unsqueeze(0).cuda()
            output = model(input)
            loss = _loss_impl(output, target)
            valid_loss.append(loss.item())
        valid_loss = torch.mean(torch.FloatTensor(valid_loss))
        # we save the model if current validation loss is less than prev : validation.pth.tar
        if (min_valid_loss == 0 or valid_loss < min_valid_loss):
            min_valid_loss = valid_loss
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'validation_loss': valid_loss
            }
            torch.save(state, modelPath + 'validation.pth.tar')
        print('epoch no ----------> {} validation loss {} '.format(epoch, valid_loss.item()))
        f.write('epoch no ----------> {} validation loss {} \n'.format(epoch, valid_loss.item()))
        epoch_loss['validation'].append(valid_loss)

    f.close()
    plotGraph(epoch_loss,basepath)

def plotGraph(epoch_loss,basepath):
    import  matplotlib.pyplot as plt
    fig = plt.figure(1)
    trainloss = epoch_loss['train']
    validloss = epoch_loss['validation']

    plt.plot(np.arange(trainloss),trainloss , 'r--',label='training loss')
    plt.plot(np.arange(validloss),validloss,'g--',label = 'validation loss')
    plt.legend()
    plt.savefig(basepath+'.png')
    plt.show()

if __name__ == "__main__":
    train('/data/Guha/GR/model/dip_nn/')




