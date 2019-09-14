from Network import BiLSTM,BiRNN
#from train_BiRNN import BiRNN

import numpy as np
from IMUDataset import  IMUDataset
import torch
import Config as cfg
import itertools
from pyquaternion import Quaternion

class TestEngine:
    def __init__(self):
        self.testSet = ['AMASS_Transition']
        self.datapath = '/data/Guha/GR/Dataset.old/'
        self.test_dataset = IMUDataset(self.datapath,self.testSet)
        self.use_cuda =True
        self.testModel = BiRNN().cuda()
        #self.testModel = BiLSTM().cuda()

        modelPath = '/data/Guha/GR/model/9/validation.pth.tar'
        self.base = '/data/Guha/GR/Output/TestSet/9/'
        with open(modelPath , 'rb') as tar:
            checkpoint = torch.load(tar)
            model_weights = checkpoint['state_dict']
            #epoch_loss = checkpoint['validation_loss']
        self.testModel.load_state_dict(model_weights)

    def test(self):
        # initialize hidden and cell state  at each new batch
        hidden = torch.zeros(cfg.n_layers * 2, 1, cfg.hid_dim, dtype=torch.double).cuda()
        cell = torch.zeros(cfg.n_layers * 2, 1, cfg.hid_dim, dtype=torch.double).cuda()
        # loop through all the files

        #for f in self.test_dataset.files:
        f = '/data/Guha/GR/Dataset/CMU/02_02_05.npz'
        self.test_dataset.readfile(f)
        input = torch.FloatTensor(self.test_dataset.input)
        input = torch.unsqueeze(input, 0)
        target = torch.FloatTensor(self.test_dataset.target)

        if self.use_cuda:
            # input = [input.cuda()]
            input = input.cuda()
            self.testModel.cuda()

        self.testModel.eval()
        # bilstm
        # prediction,_,_ = self.testModel(input,hidden,cell)

        # birnn
        prediction = self.testModel(input)

        prediction = prediction.detach().reshape_as(target).cpu()
        loss = self._loss_impl(prediction, target)
        # Renormalize prediction
        prediction = prediction.numpy().reshape(-1, 15, 4)
        seq_len = prediction.shape[0]
        norms = np.linalg.norm(prediction, axis=2)
        prediction = np.asarray(
            [prediction[k, j, :] / norms[0, 0] for k, j in itertools.product(range(seq_len), range(15))])
        # save GT and prediction
        #np.savez_compressed(self.base + f.split('/')[-1], target=target.cpu().numpy(), predictions=prediction)
        print(f, '------------', loss.item())

    def readfile(self, file):
        data_dict = np.load(file, encoding='latin1')
        sample_pose = data_dict['pose'].reshape(-1, 15, 3, 3)
        sample_ori = data_dict['ori']
#        sample_acc = data_dict['acc']
        seq_len = sample_pose.shape[0]

        #################### convert orientation matrices to quaternion ###############
        ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                               itertools.product(range(seq_len), range(5))])
        ori_quat = ori_quat.reshape(-1, 5 * 4)

        #################### convert orientation matrices to euler ###############
        # ori_euler = np.asarray([transforms3d.euler.mat2euler(sample_ori[k, j, :, :]) for k, j in
        #                         itertools.product(range(seq_len), range(5))])
        # ori_euler = ori_euler.reshape(-1, 5, 3)
        # ori_euler = ori_euler[:, :, 0:2].reshape(-1, 5 * 2)

        #################### convert pose matrices to quaternion ###############
        pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                itertools.product(range(seq_len), range(15))])

        pose_quat = pose_quat.reshape(-1, 15 * 4)
        #################### standardize acceleration #################
        ################# To normalize acceleration ###################
        # imu_dip = dict(
        #     np.load('/data/Guha/GR/code/dip18/train_and_eval/data/dipIMU/imu_own_validation.npz', encoding='latin1'))
        # data_stats = imu_dip.get('statistics').tolist()
        # acc_stats = data_stats['acceleration']
        # sample_acc = sample_acc.reshape(-1, 5 * 3)
        # sample_acc = (sample_acc - acc_stats['mean_channel']) / acc_stats['std_channel']

        #concat = np.concatenate((ori_quat, sample_acc), axis=1)

        self.input = ori_quat
        self.target = pose_quat

    def testWindow(self,len_past,len_future):
        # initialize hidden and cell state  at each new batch
        hidden = torch.zeros(cfg.n_layers * 2, 1, cfg.hid_dim, dtype=torch.double).cuda()
        cell = torch.zeros(cfg.n_layers * 2, 1, cfg.hid_dim, dtype=torch.double).cuda()
        loss_file = open('/data/Guha/GR/Output/loss_9_AMASS_Transition.txt', 'w')
        # loop through all the files
        for ct,f in enumerate(self.test_dataset.files):
            #f = '/data/Guha/GR/Dataset/DIP_IMU2/test/s_10_05.npz'
            #f = '/data/Guha/GR/Dataset.old/AMASS_Transition/mazen_c3dairkick_jumpinplace.npz'
            self.readfile(f)
            input = self.input
            target = self.target
            seq_len = input.shape[0]
            predictions = []
            # loop over all frames in input. take the window to predict each timestep t
            for step in range(seq_len):
                start_idx = max(step - len_past, 0)
                end_idx = min(step + len_future + 1, seq_len)
                in_window = input[start_idx:end_idx]
                in_window = torch.FloatTensor(in_window).unsqueeze(0).cuda()
                # target_window = target[start_idx:end_idx]

                self.testModel.eval()
                # bilstm
                #output,_,_ = self.testModel(in_window,hidden,cell)
                # birnn
                output = self.testModel(in_window)
                prediction_step = min(step, len_past)
                pred = output[:, prediction_step:prediction_step + 1].detach().cpu().numpy().reshape(15, 4)
                predictions.append(pred)

            ################## Renormalize prediction
            predictions = np.asarray(predictions)
            norms = np.linalg.norm(predictions, axis=2)
            predictions = np.asarray(
                [predictions[k, j, :] / norms[0, 0] for k, j in itertools.product(range(seq_len), range(15))])
            ##################calculate loss
            loss  = self._loss_impl(predictions,target)
            loss_file.write('{}-- {}\n'.format(f,loss))
            print(f+'-------'+str(loss))
            # save GT and prediction
            #np.savez_compressed(self.base + f.split('/')[-1], target=target, predictions=predictions)
            #print(f)
            if(ct==30):
                break
        loss_file.close()

    def _loss_impl(self, predicted, expected):
        L1 = predicted.reshape(-1,60) - expected.reshape(-1,60)
        return np.mean((np.linalg.norm(L1, 2, 1)))

if __name__ == '__main__':
    testEngine = TestEngine()
    testEngine.testWindow(20,5)
    #testEngine.test()
    #testEngine.mean_smoothing(5)


