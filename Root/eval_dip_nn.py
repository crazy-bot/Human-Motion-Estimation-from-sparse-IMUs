# from Network import BiLSTM
# from Network import BiRNN
from DIP_IMU_NN_MLP import  InverseKinematic
from DIP_IMU_NN_BiRNN import  BiRNN
import numpy as np
from IMUDataset import  IMUDataset
import torch
import Config as cfg
import itertools
import transforms3d

class TestEngine:
    def __init__(self):
        self.testModel = InverseKinematic().cuda()
        #self.testModel = BiRNN().cuda()

        modelPath = '/data/Guha/GR/model/dip_nn_mlp/validation.pth.tar'
        self.base = '/data/Guha/GR/Output/TestSet/dip_nn/'
        with open(modelPath , 'rb') as tar:
            checkpoint = torch.load(tar)
            model_weights = checkpoint['state_dict']
            #epoch_loss = checkpoint['validation_loss']
        self.testModel.load_state_dict(model_weights)
        self.loss_file = open('/data/Guha/GR/Output/loss/loss_dip_mlp.txt', 'w')

# test the whole activity at once
    def test(self):
        from pyquaternion import Quaternion
        file = '/data/Guha/GR/code/dip18/train_and_eval/data/dipIMU/imu_own_test.npz'
        data_dict = dict(np.load(file))
        for act in range(18):
            sample_pose = data_dict['smpl_pose'][act].reshape(-1, 15, 3, 3)
            sample_ori = data_dict['orientation'][act].reshape(-1, 5, 3, 3)
            act = data_dict['file_id'][act]
            seq_len = sample_ori.shape[0]
            ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                                   itertools.product(range(seq_len), range(5))])
            ori_quat = ori_quat.reshape(-1, 5 * 4)
            pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                    itertools.product(range(seq_len), range(15))])

            pose_quat = pose_quat.reshape(-1, 15, 4)

            input = torch.FloatTensor(ori_quat)
            target = pose_quat

            # bilstm
            # initialize hidden and cell state  at each new batch
            hidden = torch.zeros(cfg.n_layers * 2, 1, cfg.hid_dim, dtype=torch.double).cuda()
            cell = torch.zeros(cfg.n_layers * 2, 1, cfg.hid_dim, dtype=torch.double).cuda()
            # prediction,_,_ = self.testModel(input,hidden,cell)

            # birnn
            #prediction = self.testModel(input)

            # mlp
            prediction = self.testModel(input)
            prediction = prediction.detach().cpu().numpy().reshape(-1, 15, 4)

            # Renormalize prediction
            predictions = np.asarray(predictions)
            norms = np.linalg.norm(predictions, axis=2)
            predictions = np.asarray(
                [predictions[k, j, :] / norms[0, 0] for k, j in itertools.product(range(seq_len), range(15))])
            predictions = predictions.reshape(seq_len, 15, 4)
            ################### convert to euler
            target_euler = np.asarray([transforms3d.euler.quat2euler(target[k, j]) for k, j in
                                       itertools.product(range(seq_len), range(15))])
            target_euler = (target_euler * 180) / np.pi
            pred_euler = np.asarray([transforms3d.euler.quat2euler(predictions[k, j]) for k, j in
                                     itertools.product(range(seq_len), range(15))])
            pred_euler = (pred_euler * 180) / np.pi
            ##################calculate loss
            loss = self.loss_impl(target_euler.reshape(-1, 15, 3), pred_euler.reshape(-1, 15, 3))
            # loss_file.write('{}-- {}\n'.format(f,loss))
            # print(f+'-------'+str(loss))
            print(act + '-------' + str(loss))
            self.loss_file.write('{}\n'.format(loss))
            # save GT and prediction
            #np.savez_compressed(self.base + act , target=target.cpu().numpy(), predictions=prediction)

# test each timestep within a window
    def testWindow(self,len_past,len_future):
        from pyquaternion import Quaternion
        file = '/data/Guha/GR/code/dip18/train_and_eval/data/dipIMU/imu_own_test.npz'
        data_dict = dict(np.load(file))
        for act in range(18):
            sample_pose = data_dict['smpl_pose'][act].reshape(-1, 15, 3, 3)
            sample_ori = data_dict['orientation'][act].reshape(-1, 5, 3, 3)
            act = data_dict['file_id'][act]
            seq_len = sample_ori.shape[0]
            ori_quat = np.asarray([Quaternion(matrix=sample_ori[k, j, :, :]).elements for k, j in
                                   itertools.product(range(seq_len), range(5))])
            ori_quat = ori_quat.reshape(-1, 5 * 4)
            pose_quat = np.asarray([Quaternion(matrix=sample_pose[k, j, :, :]).elements for k, j in
                                    itertools.product(range(seq_len), range(15))])

            pose_quat = pose_quat.reshape(-1, 15 , 4)

            input = torch.FloatTensor(ori_quat)
            target = pose_quat

            # bilstm
            # initialize hidden and cell state  at each new batch
            hidden = torch.zeros(cfg.n_layers * 2, 1, cfg.hid_dim, dtype=torch.double).cuda()
            cell = torch.zeros(cfg.n_layers * 2, 1, cfg.hid_dim, dtype=torch.double).cuda()

            predictions = []
            # loop over all frames in input. take the window to predict each timestep t
            for step in range(seq_len):
                start_idx = max(step - len_past, 0)
                end_idx = min(step + len_future + 1, seq_len)
                in_window = input[start_idx:end_idx]
                in_window = torch.FloatTensor(in_window).unsqueeze(0).cuda()
                # target_window = target[start_idx:end_idx]
                # bilstm
                # output,_,_ = self.testModel(in_window,hidden,cell)

                # birnn
                output = self.testModel(in_window)

                prediction_step = min(step, len_past)
                pred = output[:, prediction_step:prediction_step + 1].detach().cpu().numpy().reshape(15, 4)
                predictions.append(pred)

            # Renormalize prediction
            predictions = np.asarray(predictions)
            norms = np.linalg.norm(predictions, axis=2)
            predictions = np.asarray(
                [predictions[k, j, :] / norms[0, 0] for k, j in itertools.product(range(seq_len), range(15))])
            predictions = predictions.reshape(seq_len, 15, 4)
            ################### convert to euler
            target_euler = np.asarray([transforms3d.euler.quat2euler(target[k, j]) for k, j in
                                       itertools.product(range(seq_len), range(15))])
            target_euler = (target_euler * 180) / np.pi
            pred_euler = np.asarray([transforms3d.euler.quat2euler(predictions[k, j]) for k, j in
                                     itertools.product(range(seq_len), range(15))])
            pred_euler = (pred_euler * 180) / np.pi
            ##################calculate loss
            loss = self.loss_impl(target_euler.reshape(-1, 15, 3), pred_euler.reshape(-1, 15, 3))
            # loss_file.write('{}-- {}\n'.format(f,loss))
            # print(f+'-------'+str(loss))
            print(act + '-------' + str(loss))
            self.loss_file.write('{}\n'.format(loss))
            # save GT and prediction
            #np.savez_compressed(self.base + act, target=target, predictions=predictions)

        self.loss_file.close()

    def loss_impl(self, predicted, expected):
        error = predicted - expected
        error_norm = np.linalg.norm(error, axis=2)
        error_per_joint = np.mean(error_norm, axis=1)
        error_per_frame_per_joint = np.mean(error_per_joint, axis=0)
        return error_per_frame_per_joint


if __name__ == '__main__':
    testEngine = TestEngine()
    testEngine.testWindow(20,5)
    #testEngine.test()



