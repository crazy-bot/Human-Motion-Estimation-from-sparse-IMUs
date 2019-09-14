from Network import InverseKinematic
import numpy as np
from IMUDataset import  IMUDataset
import torch
import Config as cfg
import itertools

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
                target = torch.FloatTensor(self.dataset.target)
                if self.use_cuda:
                    input = input.cuda()

                prediction = self.testModel(input)
                prediction = prediction.detach().reshape_as(target).cpu()
                #loss = torch.norm((prediction-target), 2, 1)
                loss = self._loss_impl(prediction, target)
                loss_file.write('{}-- {}\n'.format(file, loss))
                # mean_loss = torch.mean(loss)
                # max_loss = torch.max(loss)
                # dset_loss.extend(loss.numpy())
                # valid_meanloss.append(mean_loss)
                # valid_maxloss.append(max_loss)
                # print('mean loss {},  max loss {}, \n'.format(mean_loss, max_loss))
                # np.savez_compressed(self.base + file.split('/')[-1], target=target.cpu().numpy(), predictions=prediction)

        loss_file.close()

    def _loss_impl(self, predicted, expected):
        L1 = predicted - expected
        return torch.mean((torch.norm(L1, 2, 1)))

if __name__ == '__main__':
    testEngine = TestEngine()
    testEngine.test()



