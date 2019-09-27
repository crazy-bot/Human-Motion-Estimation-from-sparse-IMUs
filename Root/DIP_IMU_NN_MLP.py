import torch
import torch.nn as nn
import torch.optim as optim
import Config as cfg
import numpy as np
from pyquaternion import Quaternion
import  itertools
import random

class InverseKinematic(nn.Module):
    def __init__(self):
        super(InverseKinematic,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(20,60), nn.ReLU(),nn.Linear(60,180),nn.ReLU(),nn.Linear(180,60)
        )
    def forward(self, input):
        out = self.net(input)
        return out


def _loss_impl(predicted, expected):
    L1 = predicted - expected
    return torch.mean((torch.norm(L1, 2, 1)))

def preparedata(path):
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

        batch_ori.append(ori_quat)
        batch_pose.append(pose_quat)

    return batch_ori,batch_pose

def train(basepath):

    model = InverseKinematic().cuda()
    modelPath = basepath
    optimizer = optim.Adam(model.parameters(), lr=.001)
    trainpath = '/data/Guha/GR/DIPIMUandOthers/DIP_IMU_and_Others/DIP_IMU_nn/imu_own_training.npz'
    validpath = '/data/Guha/GR/DIPIMUandOthers/DIP_IMU_and_Others/DIP_IMU_nn/imu_own_validation.npz'
    # trainpath = '/data/Guha/GR/dataset/DIP_IMU_nn/imu_own_training.npz'
    # validpath = '/data/Guha/GR/dataset/DIP_IMU_nn/imu_own_validation.npz'
    f = open(modelPath + 'model_details', 'w')
    f.write(' comments: dip_imu_nn training on MLP')
    f.write(str(model))
    f.write('\n')
    epoch_loss = {'train': [],'validation':[]}
    batch_ori, batch_pose = preparedata(trainpath)

    print('no of batches--', len(batch_ori))
    f.write('no of batches-- {} \n'.format(len(batch_ori)))

    min_valid_loss = 0.0
    for epoch in range(50):
        ############### training #############
        train_loss = []
        model.train()
        for input,target in zip(batch_ori,batch_pose):
            input = torch.FloatTensor(input).cuda()
            target = torch.FloatTensor(target).cuda()
            output = model(input)
            loss = _loss_impl(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss.detach()
            train_loss.append(loss.item())

        train_loss = torch.mean(torch.FloatTensor(train_loss))
        print('epoch no ----------> {} training loss {} '.format(epoch, train_loss.item()))
        f.write('epoch no ----------> {} training loss {} '.format(epoch, train_loss.item()))
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

            input = torch.FloatTensor(ori_quat).cuda()
            target = torch.FloatTensor(pose_quat).cuda()
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
        f.write('epoch no ----------> {} validation loss {} '.format(epoch, valid_loss.item()))
        epoch_loss['validation'].append(valid_loss)

    f.close()
    plotGraph(epoch_loss,basepath)

def plotGraph(epoch_loss,basepath):
    import  matplotlib.pyplot as plt
    fig = plt.figure(1)
    trainloss = epoch_loss['train']
    validloss = epoch_loss['validation']

    plt.plot(np.arange(len(trainloss)),trainloss , 'r--',label='training loss')
    plt.plot(np.arange(len(validloss)),validloss,'g--',label = 'validation loss')
    plt.legend()
    plt.savefig(basepath+'.png')
    plt.show()

if __name__ == "__main__":
    train('/data/Guha/GR/model/dip_nn_mlp/')




