import numpy as np
from Network import ForwardKinematic
import torch.optim as optim
from IMUDataset import IMUDataset
from time import time
import torch
import Config as cfg
import torch.nn as nn
import random
import matplotlib.pyplot as plt

class TrainingEngine:
    def __init__(self):
        # self.trainset = ['AMASS_ACCAD', 'AMASS_BioMotion', 'AMASS_CMU_Kitchen', 'AMASS_Eyes', 'AMASS_MIXAMO',
        #                  'AMASS_SSM', 'AMASS_Transition', 'CMU', 'H36']
        self.trainset = ['H36','DIP_IMU/train']
        #self.testSet = ['AMASS_HDM05', 'HEva', 'JointLimit']
        self.testset = ['DIP_IMU/validation']
        self.dataPath = '/data/Guha/GR/Dataset/'
        self.train_dataset = IMUDataset(self.dataPath, self.trainset)
        self.valid_dataset = IMUDataset(self.dataPath, self.testset)
        self.modelPath = '/data/Guha/GR/model/forward/'
        self.model = ForwardKinematic().cuda()
        self.mseloss = nn.MSELoss(reduction='sum')
        # baseModelPath = '/data/Guha/GR/model/16/validation.pth.tar'
        #
        # with open(baseModelPath, 'rb') as tar:
        #     checkpoint = torch.load(tar)
        #     model_weights = checkpoint['state_dict']
        # self.model.load_state_dict(model_weights)

    def train(self,n_epochs):
        f = open(self.modelPath+'model_details','a')
        f.write(str(self.model))
        f.write('\n')

        np.random.seed(1234)
        lr = 0.001
        optimizer = optim.Adam(self.model.parameters(),lr=lr)

        print('Training for %d epochs' % (n_epochs))
        no_of_trainbatch = int(self.train_dataset.total_frames / 200)

        print('batch size--> {},  no of batches--> {}'.format(200, no_of_trainbatch))
        f.write('batch size--> {}, no of batches--> {} \n'.format(200, no_of_trainbatch))

        min_valid_loss = 0.0

        try:
            ################ epoch loop ###################
            epoch_loss = {'train': [], 'validation': []}
            for epoch in range(n_epochs):
                train_loss = []
                start_time = time()
                ####################### training #######################
                self.train_dataset.loadfiles(self.dataPath, self.trainset)
                self.model.train()
                while(len(self.train_dataset.files) > 0):
                    # Pick a random chunk from each sequence
                    self.train_dataset.createbatch_no_replacement()
                    outputs = self.train_dataset.input
                    inputs = self.train_dataset.target
                    data = [(inputs[i], outputs[i]) for i in range(len(inputs))]
                    random.shuffle(data)
                    X, Y = zip(*data)
                    X = torch.FloatTensor(list(X)).cuda()
                    Y = torch.FloatTensor(list(Y)).cuda()
                    X_list = list(torch.split(X, 200))
                    Y_list = list(torch.split(Y, 200))
                    for x,y in zip(X_list,Y_list):

                        optimizer.zero_grad()
                        predictions = self.model(x)

                        loss = self._loss_impl(predictions, y)
                        loss.backward()
                        optimizer.step()
                        loss.detach()
                        train_loss.append(loss.item())

                train_loss = torch.mean(torch.FloatTensor(train_loss))
                epoch_loss['train'].append(train_loss)
                # we save the model after each epoch : epoch_{}.pth.tar
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'epoch_loss': epoch_loss
                }
                torch.save(state, self.modelPath + 'epoch_{}.pth.tar'.format(epoch+1))
                debug_string = 'epoch No {}, epoch loss {}, Time taken {} \n'.format(
                    epoch + 1, train_loss, start_time - time()
                )
                print(debug_string)
                f.write(debug_string)
                f.write('\n')
                ####################### Validation #######################
                self.model.eval()
                self.valid_dataset.loadfiles(self.dataPath, self.testset)
                valid_loss = []
                for file in self.valid_dataset.files:
                    self.valid_dataset.readfile(file)
                    target = torch.FloatTensor(self.valid_dataset.input)
                    input = torch.FloatTensor(self.valid_dataset.target)
                    input = input.cuda()

                    prediction = self.model(input)
                    prediction = prediction.detach().reshape_as(target).cpu()
                    loss = self._loss_impl(prediction,target)
                    # loss = (prediction - target)
                    valid_loss.append(loss.item())

                valid_loss = torch.mean(torch.FloatTensor(valid_loss))
                epoch_loss['validation'].append(valid_loss)
                # we save the model if current validation loss is less than prev : validation.pth.tar
                if (min_valid_loss == 0 or valid_loss < min_valid_loss):
                    min_valid_loss = valid_loss
                    state = {
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'validation_loss': valid_loss
                    }
                    torch.save(state, self.modelPath + 'validation.pth.tar')


                # save box plots of three dataset
                # fig = plt.figure('epoch: '+str(epoch))
                # # Create an axes instance
                # ax = fig.add_subplot(111)
                # # Create the boxplot
                # ax.boxplot(dset_loss)
                # ax.set_xticklabels(self.testSet)
                # # Save the figure
                # fig.savefig(self.modelPath+'epoch: '+str(epoch)+'.png', bbox_inches='tight')

                # logging to track
                debug_string = 'epoch No {}, validation loss {}, Time taken {} \n'.format(
                    epoch + 1, valid_loss, start_time - time()
                )
                print(debug_string)
                f.write(debug_string)
                f.write('\n')

            f.write('{}'.format(epoch_loss))
            f.close()
        except KeyboardInterrupt:
            print('Training aborted.')

    def _loss_impl(self, predicted, expected):
        L1 = predicted - expected
        return torch.mean((torch.norm(L1, 2, 1)))


if __name__ == '__main__':
    trainingEngine = TrainingEngine()
    trainingEngine.train(n_epochs=30)