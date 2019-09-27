from Network import InverseKinematic
import numpy as np
from IMUDataset import  IMUDataset
import torch
import Config as cfg
import itertools
import  transforms3d

class TestEngine:
    def __init__(self):
        self.testSet = ['s_11']
        self.datapath = '/data/Guha/GR/Dataset.old/'
        self.dataset = IMUDataset(self.datapath,self.testSet)
        self.use_cuda =True
        self.testModel = InverseKinematic().cuda()

        modelPath = '/data/Guha/GR/model/14/validation.pth.tar'
        self.base = '/data/Guha/GR/Output/TestSet/14/'
        with open(modelPath , 'rb') as tar:
            checkpoint = torch.load(tar)
            model_weights = checkpoint['state_dict']
        self.testModel.load_state_dict(model_weights)

    def test(self):
        valid_meanloss = []
        valid_maxloss = []
        # data_to_plot = []
        self.testModel.eval()
        loss_file = open('/data/Guha/GR/Output/loss_14.txt', 'w')
        for d in self.testSet:
            self.dataset.loadfiles(self.datapath, [d])
            dset_loss = []
            for file in self.dataset.files:
                #file = '/data/Guha/GR/Dataset.old/s_11/S11_WalkTogether.npz'
                self.dataset.readfile(file)
                input = torch.FloatTensor(self.dataset.input)
                seq_len = len(input)
                target = torch.FloatTensor(self.dataset.target)
                target = target.reshape(seq_len,15,4)
                if self.use_cuda:
                    input = input.cuda()

                predictions = self.testModel(input)

                ################## Renormalize prediction
                predictions = predictions.detach().reshape_as(target).cpu()
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


                loss = self.loss_impl(pred_euler.reshape(-1,15,3), target_euler.reshape(-1,15,3))
                loss_file.write('{}\n'.format(loss))
                # mean_loss = torch.mean(loss)
                # max_loss = torch.max(loss)
                # dset_loss.extend(loss.numpy())
                # valid_meanloss.append(mean_loss)
                # valid_maxloss.append(max_loss)
                # print('mean loss {},  max loss {}, \n'.format(mean_loss, max_loss))
                # np.savez_compressed(self.base + file.split('/')[-1], target=target.cpu().numpy(), predictions=prediction)

        loss_file.close()

    def loss_impl(self, predicted, expected):
        error = predicted - expected
        error_norm = np.linalg.norm(error, axis=2)
        error_per_joint = np.mean(error_norm, axis=1)
        error_per_frame_per_joint = np.mean(error_per_joint, axis=0)
        return error_per_frame_per_joint

if __name__ == '__main__':
    testEngine = TestEngine()
    testEngine.test()



