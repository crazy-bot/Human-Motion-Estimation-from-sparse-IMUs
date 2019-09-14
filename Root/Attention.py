import torch.nn as nn
import  torch
import os
import numpy as np
from pyquaternion import Quaternion
import itertools

class Encoder(nn.Module):
    def __init__(self,  input_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.gru = nn.GRU(input_dim, enc_units,bidirectional=False)

    def forward(self, enc_input):
        # x: batch_size,seq_len,input_dim
        self.batch_sz = enc_input.shape[0]
        # x transformed = seq_len X batch_size X input_dim
        enc_input = enc_input.permute(1, 0, 2)
        self.hidden = self.initialize_hidden_state()

        # output: seq_len, batch, num_directions * enc_units
        # self.hidden: num_layers * num_directions, batch, enc_units
        output, self.hidden = self.gru(enc_input, self.hidden) # gru returns hidden state of all timesteps as well as hidden state at last timestep

        # outputs are always from the top hidden layer. self.hidden is hidden state for t = seq_len
        return output, self.hidden

    def initialize_hidden_state(self):
        # h_0  of shape(num_layers * num_directions, batch, enc_units)
        return torch.zeros((1*1 , self.batch_sz, self.enc_units)).cuda()


class Decoder(nn.Module):
    def __init__(self, output_dim, dec_units, enc_units):
        super(Decoder, self).__init__()

        self.dec_units = dec_units
        self.enc_units = enc_units

        self.output_dim = output_dim

        self.gru = nn.GRU(self.output_dim + self.enc_units,self.dec_units,batch_first=True)
        self.fc = nn.Linear(self.dec_units, self.output_dim)

        #dec_units, enc_units : hidden units for encoder and decoder
        # used for attention
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.dec_units, 1)

    def forward(self, dec_input, enc_hidden, enc_output):
        # enc_output original: seq_len, batch, num_directions * hidden_size
        # enc_output converted == batch, seq_len, num_directions * hidden_size
        enc_output = enc_output.permute(1, 0, 2)
        self.batch_sz = enc_output.shape[0]
        # hidden shape == num_layers * num_directions, batch, hidden_size
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = enc_hidden.permute(1, 0, 2)

        # score: (batch_size, seq_len, hidden_size) # Bahdanaus's
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        # It doesn't matter which FC we pick for each of the inputs
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, seq_len, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = torch.softmax(self.V(score), dim=1)

        # context_vector shape after sum == (batch_size, enc_units)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)

        # dec_input shape == (batch_size, 1, output_dim)

        # dec_input shape after concatenation == (batch_size, 1, output_dim + enc_units)
        dec_input = torch.cat((context_vector.unsqueeze(1), dec_input), -1)

        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, dec_units)
        output, state = self.gru(dec_input)

        # output shape == (batch_size * 1, dec_units)
        output = output.view(-1, output.size(2))

        # output shape == (batch_size * 1, output_dim)
        x = self.fc(output)

        return x, state

    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.dec_units))

class RawDataset():

    def __init__(self):
       return

    def loadfiles(self, datapath, trainset):
        listofPath = []

        for d in trainset:
            folderpath = os.path.join(datapath, d)
            for f in os.listdir(folderpath):
                listofPath.append(os.path.join(folderpath, f))
        self.files = listofPath
        return listofPath

    def createbatch_no_replacement(self, chunk_sz):
        idx = np.random.choice(len(self.files))
        print('reading file--',self.files[idx])
        data_dict = np.load(self.files.pop(idx), encoding='latin1')
        sample_pose = np.asarray(data_dict['poses'])
        sample_ori = np.asarray(data_dict['ori'])
        sample_pose = sample_pose.reshape(-1, 15, 3, 3)
        sample_ori = sample_ori.reshape(-1, 6, 3, 3)
        if(len(sample_pose) == 0):
            return [],[]

        #################### convert all roation matrices to quaternion ###############
        ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                               itertools.product(range(sample_ori.shape[0]), range(6))])
        ori_quat = ori_quat.reshape(-1, 6 * 4)
        pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                itertools.product(range(sample_pose.shape[0]), range(15))])
        pose_quat = pose_quat.reshape(-1, 15 * 4)
        inputs = torch.FloatTensor(ori_quat)
        outputs = torch.FloatTensor(pose_quat)

        if (inputs.size(0) == chunk_sz):
            chunk_in = list(torch.split(inputs, chunk_sz))
            chunk_out = list(torch.split(outputs, chunk_sz))
            chunk_in = torch.stack(chunk_in, dim=0).cuda()
            chunk_out = torch.stack(chunk_out, dim=0).cuda()
        elif(inputs.size(0) > chunk_sz):
            chunk_in = list(torch.split(inputs, chunk_sz))[:-1]
            chunk_out = list(torch.split(outputs,chunk_sz))[:-1]
            chunk_in = torch.stack(chunk_in, dim=0).cuda()
            chunk_out = torch.stack(chunk_out, dim=0).cuda()
        else:
            chunk_in = inputs.unsqueeze(0).cuda()
            chunk_out = outputs.unsqueeze(0).cuda()

        return chunk_in,chunk_out

    def prepareBatchOfMotion(self,batch_sz):
        inputs = []
        targets = []
        self.input = []
        self.target = []
        # from sorted files pick sequentiallly
        for i in range(batch_sz):
            if(len(self.files) == 0):
                break
            idx = np.random.choice(len(self.files))
            print('reading file--', self.files[idx])

            data_dict = np.load(self.files.pop(idx), encoding='latin1')
            sample_pose = np.array(data_dict['poses']).reshape(-1, 15, 3, 3)
            sample_ori = np.array(data_dict['ori']).reshape(-1, 6, 3, 3)
            seq_len = sample_pose.shape[0]

            #################### convert orientation matrices to Quat ###############
            ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                                   itertools.product(range(seq_len), range(6))])
            ori_quat = ori_quat.reshape(-1, 6 * 4)

            #################### convert pose matrices to quaternion #################
            pose = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                    itertools.product(range(seq_len), range(15))])
            pose = pose.reshape(-1, 15 * 4)
            inputs.append(ori_quat)
            targets.append(pose)

        # padding of input and output to make the batch of same sequence length
        max_len = max([pose.shape[0] for pose in targets])
        for input,target in zip(inputs,targets):
            quat_ori = np.repeat(Quaternion(matrix=np.eye(3,3)).elements[np.newaxis, ...], 6, axis=0)
            ori = np.array([quat_ori]*max_len).reshape(max_len,-1)
            seq_len = input.shape[0]
            ori[:seq_len, :] = input
            padded_in = torch.Tensor(ori).cuda()

            quat_pose = np.repeat(Quaternion(matrix=np.eye(3, 3)).elements[np.newaxis, ...], 15, axis=0)
            pose = np.array([quat_pose] * max_len).reshape(max_len,-1)
            pose[:seq_len, :] = target
            padded_pose = torch.Tensor(pose).cuda()
            self.input.append(padded_in)
            self.target.append(padded_pose)

        if(len(self.input) > 0):
            self.input = torch.stack(self.input)
            self.target = torch.stack(self.target)

    ############## below method should be called during testing #################
    def readfile(self,file,chunk_sz):
        print('reading file--', file)
        data_dict = np.load(file, encoding='latin1')
        sample_pose = np.asarray(data_dict['poses'])
        sample_ori = np.asarray(data_dict['ori'])
        sample_pose = sample_pose.reshape(-1, 15, 3, 3)
        sample_ori = sample_ori.reshape(-1, 6, 3, 3)
        if (len(sample_pose) == 0):
            return [], []

        #################### convert all roation matrices to quaternion ###############
        ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                               itertools.product(range(sample_ori.shape[0]), range(6))])
        ori_quat = ori_quat.reshape(-1, 6 * 4)
        pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                itertools.product(range(sample_pose.shape[0]), range(15))])
        pose_quat = pose_quat.reshape(-1, 15 * 4)
        inputs = torch.FloatTensor(ori_quat).cuda()
        outputs = torch.FloatTensor(pose_quat).cuda()
        chunk_in = list(torch.split(inputs, chunk_sz))
        chunk_out = list(torch.split(outputs, chunk_sz))

        return chunk_in, chunk_out

    def readDFKIfile(self, file, chunk_sz):
        print('reading file--', file)
        data_dict = np.load(file, encoding='latin1')
        sample_ori = np.asarray(data_dict['ori'])

        if (len(sample_ori) == 0):
            return [], []

        #################### convert all roation matrices to quaternion ###############

        ori_quat = sample_ori.reshape(-1, 6 * 4)
        inputs = torch.FloatTensor(ori_quat).cuda()
        chunk_in = list(torch.split(inputs, chunk_sz))

        return chunk_in

    def readDIPfile(self,file,chunk_sz):
        import quaternion
        print('reading file--', file)
        sensor_idx = [7, 8, 11, 12, 0, 2]
        data_dict = np.load(file, encoding='latin1')
        imu_ori = data_dict['imu'][:, :, 0:9]
        frames2del = np.unique(np.where(np.isnan(imu_ori) == True)[0])
        imu_ori = np.delete(imu_ori, frames2del, 0)
        imu_ori = np.asarray([imu_ori[:, k, :] for k in sensor_idx])
        sample_ori = imu_ori.reshape(-1, 6, 3, 3)

        imu_pose = data_dict['gt']
        imu_pose = np.delete(imu_pose, frames2del, 0)
        SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
        imu_pose = imu_pose.reshape(-1,24,3)[:, SMPL_MAJOR_JOINTS, :]
        qs = quaternion.from_rotation_vector(imu_pose)
        pose_rot = np.reshape(quaternion.as_rotation_matrix(qs), [imu_pose.shape[0], 15, 9])
        sample_pose = pose_rot.reshape(-1, 15, 3, 3)


        #################### convert all roation matrices to quaternion ###############
        ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                               itertools.product(range(sample_ori.shape[0]), range(6))])
        ori_quat = ori_quat.reshape(-1, 6 * 4)
        pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                itertools.product(range(sample_pose.shape[0]), range(15))])
        pose_quat = pose_quat.reshape(-1, 15 * 4)
        inputs = torch.FloatTensor(ori_quat).cuda()
        outputs = torch.FloatTensor(pose_quat).cuda()
        chunk_in = list(torch.split(inputs, chunk_sz))
        chunk_out = list(torch.split(outputs, chunk_sz))

        return chunk_in, chunk_out


