import numpy as np
from Network import BiRNN
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
        self.trainset = ['AMASS_ACCAD', 'AMASS_BioMotion', 'AMASS_CMU_Kitchen','CMU', 'HEva']
        self.testSet = ['H36']
        self.datapath = '/data/Guha/GR/Dataset'
        self.dataset = IMUDataset(self.datapath,self.trainset)
        self.use_cuda =True
        self.modelPath = '/data/Guha/GR/model/17/'
        self.model = BiRNN().cuda()
        self.mseloss = nn.MSELoss()
        # baseModelPath = '/data/Guha/GR/model/13/epoch_5.pth.tar'
        #
        # with open(baseModelPath, 'rb') as tar:
        #     checkpoint = torch.load(tar)
        #     model_weights = checkpoint['state_dict']
        # self.model.load_state_dict(model_weights)

    def train(self,n_epochs):
        f = open(self.modelPath+'model_details','w')
        f.write(str(self.model))
        f.write('\n')

        np.random.seed(1234)
        lr = 0.001
        gradient_clip = 0.1
        optimizer = optim.Adam(self.model.parameters(),lr=lr)

        print('Training for %d epochs' % (n_epochs))
        no_of_trainbatch = int(len(self.dataset.files) / cfg.batch_len)

        print('batch size--> %d, Seq len--> %d, no of batches--> %d' % (cfg.batch_len, cfg.seq_len, no_of_trainbatch))
        f.write('batch size--> %d, Seq len--> %d, no of batches--> %d \n' % (cfg.batch_len, cfg.seq_len, no_of_trainbatch))

        min_batch_loss = 0.0
        min_valid_loss = 0.0

        try:
            for epoch in range(6,n_epochs):
                epoch_loss = []
                start_time = time()
                self.dataset.loadfiles(self.datapath,self.trainset)
                ####################### training #######################
                while(len(self.dataset.files) > 0):
                    # Pick a random chunk from each sequence
                    self.dataset.createbatch_no_replacement()
                    inputs = torch.FloatTensor(self.dataset.input)
                    outputs = torch.FloatTensor(self.dataset.target)

                    if self.use_cuda:
                        inputs = inputs.cuda()
                        outputs = outputs.cuda()
                        self.model.cuda()

                    chunk_in = list(torch.split(inputs, cfg.seq_len))[:-1]
                    chunk_out = list(torch.split(outputs, cfg.seq_len))[:-1]
                    if(len(chunk_in) == 0):
                        continue
                    data = [(chunk_in[i], chunk_out[i]) for i in range(len(chunk_in))]
                    random.shuffle(data)
                    X, Y = zip(*data)
                    chunk_in = torch.stack(X, dim=0)
                    chunk_out = torch.stack(Y, dim=0)
                    print('no of chunks %d \n' % (len(chunk_in)))
                    f.write('no of chunks %d  \n' % (len(chunk_in)))
                    self.model.train()
                    optimizer.zero_grad()
                    predictions = self.model(chunk_in)

                    loss = self._loss_impl(predictions, chunk_out)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                    optimizer.step()
                    loss.detach()

                    epoch_loss.append(loss.item())
                    if (min_batch_loss == 0 or loss < min_batch_loss):
                        min_batch_loss = loss
                        print ('training loss %f ' % (loss.item()))
                        f.write('training loss %f \n' % (loss.item()))

                epoch_loss = torch.mean(torch.FloatTensor(epoch_loss))
                # we save the model after each epoch : epoch_{}.pth.tar
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'epoch_loss': epoch_loss
                }
                torch.save(state, self.modelPath + 'epoch_{}.pth.tar'.format(epoch+1))

                ####################### Validation #######################
                valid_meanloss = []
                valid_maxloss = []
                data_to_plot = []
                self.model.eval()

                for d in self.testSet:
                    self.dataset.loadfiles(self.datapath, [d])
                    dset_loss = []
                    for file in self.dataset.files:
                        self.dataset.readfile(file)
                        input = torch.FloatTensor(self.dataset.input)
                        input = torch.unsqueeze(input, 0)
                        target = torch.FloatTensor(self.dataset.target)
                        if self.use_cuda:
                            input = input.cuda()

                        prediction = self.model(input)
                        prediction = prediction.detach().reshape_as(target).cpu()
                        #loss = torch.norm((prediction-target),2,1)
                        loss = (prediction - target)
                        mean_loss = torch.mean(loss)
                        max_loss = torch.max(loss)
                        dset_loss.extend(loss.numpy())
                        valid_meanloss.append(mean_loss)
                        valid_maxloss.append(max_loss)
                        print(
                            'mean loss %f,  max loss %f, \n' % (
                                 mean_loss, max_loss))

                    data_to_plot.append(np.asarray(dset_loss).reshape(-1))

                # save box plots of three dataset
                fig = plt.figure('epoch: '+str(epoch))
                # Create an axes instance
                ax = fig.add_subplot(111)
                # Create the boxplot
                ax.boxplot(data_to_plot)
                ax.set_xticklabels(self.testSet)
                # Save the figure
                fig.savefig(self.modelPath+'epoch: '+str(epoch)+'.png', bbox_inches='tight')

                mean_valid_loss = torch.mean(torch.FloatTensor(valid_meanloss))
                max_valid_loss = torch.max(torch.FloatTensor(valid_maxloss))
                # we save the model if current validation loss is less than prev : validation.pth.tar
                if (min_valid_loss == 0 or mean_valid_loss < min_valid_loss):
                    min_valid_loss = mean_valid_loss
                    state = {
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'validation_loss': mean_valid_loss
                    }
                    torch.save(state, self.modelPath + 'validation.pth.tar')

                # logging to track
                debug_string = 'epoch No {}, epoch loss {} , validation mean loss {}, validation max loss {}, Time taken {} \n'.format(
                    epoch + 1, epoch_loss, mean_valid_loss.item(), max_valid_loss.item(), start_time - time()
                )
                print(debug_string)
                f.write(debug_string)

            f.close()
        except KeyboardInterrupt:
            print('Training aborted.')

    def _loss_impl(self, predicted, expected):
        L1 = predicted - expected
        batch_size = predicted.shape[0]
        dist = torch.sum(torch.norm(L1, 2, 2))
        return  dist/ batch_size



if __name__ == '__main__':
    trainingEngine = TrainingEngine()
    trainingEngine.train(n_epochs=30)