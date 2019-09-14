from Network import BiLSTM
from Network import ForwardKinematic,InverseKinematic
from train_BiRNN import BiRNN
import numpy as np
from IMUDataset import  IMUDataset
import torch
import Config as cfg
import itertools
from pyquaternion import Quaternion

class TestEngine:
    def __init__(self):
        self.use_cuda =True
        self.poseModel = BiRNN().cuda()
        #self.testModel = BiLSTM()
        self.oriModel = ForwardKinematic().cuda()

        oriModelPath = '/data/Guha/GR/model/forward/validation.pth.tar'
        poseModelPath = '/data/Guha/GR/model/H36_DIP/validation.pth.tar'
        self.base = '/data/Guha/GR/Output/TestSet/h36_dip/'

        with open(oriModelPath , 'rb') as tar:
            checkpoint = torch.load(tar)
            model_weights = checkpoint['state_dict']
            #epoch_loss = checkpoint['validation_loss']
            self.oriModel.load_state_dict(model_weights)
            self.oriModel.eval()
        with open(poseModelPath, 'rb') as tar:
            checkpoint = torch.load(tar)
            model_weights = checkpoint['state_dict']
            # epoch_loss = checkpoint['validation_loss']
            self.poseModel.load_state_dict(model_weights)
            self.poseModel.eval()

        self.mse = torch.nn.MSELoss()

    def test(self):
        # initialize hidden and cell state  at each new batch
        # hidden = torch.zeros(cfg.n_layers * 2, 1, cfg.hid_dim, dtype=torch.double).cuda()
        # cell = torch.zeros(cfg.n_layers * 2, 1, cfg.hid_dim, dtype=torch.double).cuda()
        # loop through all the files
        #valid_loss = []
        import os
        for file in os.listdir('/data/Guha/GR/Dataset/s_11'):

            file = os.path.join('/data/Guha/GR/Dataset/s_11',file)
            data_dict = np.load(file, encoding='latin1')
            sample_ori = data_dict['ori'].reshape(-1,5,3,3)
            sample_pose = data_dict['pose'].reshape(-1,15,3,3)
            seq_len = sample_ori.shape[0]

            #################### convert orientation matrices to quaternion ###############
            ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                                   itertools.product(range(seq_len), range(5))])
            ori_quat = ori_quat.reshape(-1, 5 * 4)
            pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                   itertools.product(range(seq_len), range(15))])
            pose_quat = pose_quat.reshape(-1, 15 * 4)

            input = torch.FloatTensor(pose_quat).cuda()
            #input = torch.unsqueeze(input, 0)

            # bilstm
            # prediction,_,_ = self.testModel(input,hidden,cell)
            # birnn
            pred_ori = self.oriModel(input)
            pred_pose = self.poseModel(pred_ori)
            pred_ori = pred_ori.reshape(-1,5*4)
            # loss = self.mse(pred_ori,ori_quat)
            # print('loss between calculated ori and predicted ori---',loss.item())

            prediction = pred_pose.detach().reshape(-1,15,4).cpu()
            # Renormalize prediction
            prediction = prediction.numpy()
            norms = np.linalg.norm(prediction, axis=2)
            prediction = np.asarray(
                [prediction[k, j, :] / norms[0, 0] for k, j in itertools.product(range(seq_len), range(15))])
            # save GT and prediction
            np.savez_compressed(self.base + file.split('/')[-1],target=pose_quat, predictions=prediction)

    def testWindow(self,len_past,len_future):
        # initialize hidden and cell state  at each new batch
        hidden = torch.zeros(cfg.n_layers * 2, 1, cfg.hid_dim, dtype=torch.double).cuda()
        cell = torch.zeros(cfg.n_layers * 2, 1, cfg.hid_dim, dtype=torch.double).cuda()
        # loop through all the files
        datapath = '/data/Guha/GR/Dataset/DIP_IMU/test/'
        import os
        for file in os.listdir(datapath):

            file = os.path.join(datapath, file)
            data_dict = np.load(file, encoding='latin1')
            sample_ori = data_dict['ori'].reshape(-1, 5, 3, 3)
            sample_pose = data_dict['pose'].reshape(-1, 15, 3, 3)
            seq_len = sample_ori.shape[0]

            #################### convert orientation matrices to quaternion ###############
            ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                                   itertools.product(range(seq_len), range(5))])
            ori_quat = ori_quat.reshape(-1, 5 * 4)
            pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                    itertools.product(range(seq_len), range(15))])
            pose_quat = pose_quat.reshape(-1, 15 * 4)

            input = torch.FloatTensor(pose_quat).cuda()
            #input = torch.unsqueeze(input, 0)

            pred_ori = self.oriModel(input)
            # # Renormalize prediction
            pred_ori = pred_ori.detach().cpu().numpy().reshape(-1, 5, 4)
            norms = np.linalg.norm(pred_ori,axis=2)
            pred_ori = np.asarray([pred_ori[k,j,:]/norms[0,0] for k,j in itertools.product(range(seq_len),range(5))])
            pred_ori = pred_ori.reshape(-1,20)

            # self.plotJointError(predictions,ori_quat,file.split('/')[-1])
            # continue
            #loop over all frames in input. take the window to predict each timestep t
            predictions = []
            for step in range(seq_len):
                start_idx = max(step - len_past, 0)
                end_idx = min(step + len_future + 1, seq_len)
                in_window = pred_ori[start_idx:end_idx]
                in_window = torch.cuda.FloatTensor(in_window).unsqueeze(0)
                #target_window = target[start_idx:end_idx]
                output= self.poseModel(in_window)
                prediction_step = min(step, len_past)
                pred = output[:, prediction_step:prediction_step + 1].detach().cpu().numpy().reshape(15, 4)
                predictions.append(pred)

            # Renormalize prediction
            predictions = np.asarray(predictions)
            norms = np.linalg.norm(predictions,axis=2)
            predictions = np.asarray([predictions[k,j,:]/norms[0,0] for k,j in itertools.product(range(seq_len),range(15))])
            # save GT and prediction
            np.savez_compressed(self.base+file.split('/')[-1],target=pose_quat,predictions=predictions)
            print (file)

    def _loss_impl(self, predicted, expected):
        predicted = predicted.reshape(1,-1,60)
        expected = expected.reshape(1,-1,60)
        L1 = (predicted - expected)
        seq_len = predicted.shape[1]
        dist = torch.sum(torch.norm(L1, 2, 2)) / seq_len
        return  dist*200

    def plotJointError(self,pred,target,act):
        SENSORS = ['lelbow', 'relbow', 'lknee', 'rknee', 'head', 'belly']
        import matplotlib.pyplot as plt
        import myUtil
        pred = pred.reshape(-1, 5, 4)
        target = target.reshape(-1,5, 4)
        target_aa = myUtil.quat_to_aa_representation(target, 5).reshape(-1, 5, 3)
        pred_aa = myUtil.quat_to_aa_representation(pred, 5).reshape(-1, 5, 3)
        err = (pred - target)
        err_aa = myUtil.quat_to_aa_representation(pred, 5).reshape(-1,5,3)
        colors = ['red', 'black', 'blue']
        #colors = np.asarray(colors).reshape(-1,4)
        for i in range(5):
            plt.figure()
            for j, color in enumerate(colors):
                plt.plot(np.arange(len(err)), pred_aa[:,i,j],c=color)
                plt.plot(np.arange(len(err)), target_aa[:, i, j],c=color,dashes=[4, 2])
            plt.title(act+' : '+SENSORS[i])
            plt.show()
            plt.savefig('/data/Guha/GR/Output/graphs/ori_err/'+act+' : '+SENSORS[i]+'.png')


if __name__ == '__main__':
    testEngine = TestEngine()
    testEngine.testWindow(20,5)
    #testEngine.test()
    #testEngine.mean_smoothing(5)


