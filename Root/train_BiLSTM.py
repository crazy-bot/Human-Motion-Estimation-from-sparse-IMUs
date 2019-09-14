import numpy as np
from Network import BiLSTM
import torch.optim as optim
from IMUDataset import  IMUDataset
from time import time
import torch
import Config as cfg
import torch.nn as nn

class TrainingEngine:
    def __init__(self):
        self.dataPath = '/data/Guha/GR/Dataset.old/'
        self.trainset = ['s1_s9']
        self.testset = ['s_11']
        self.train_dataset = IMUDataset(self.dataPath, self.trainset)
        self.valid_dataset = IMUDataset(self.dataPath, self.testset)
        self.modelPath = '/data/Guha/GR/model/sequential/'
        self.model = BiLSTM().cuda()
        #self.mseloss = nn.MSELoss(reduction='sum')

    def train(self,n_epochs):

        f = open(self.modelPath+'model_details','w')
        f.write(str(self.model))
        f.write('\n')

        np.random.seed(1234)
        lr = 0.001
        gradient_clip = 0.1
        optimizer = optim.Adam(self.model.parameters(),lr=lr)

        print('Training for %d epochs' % (n_epochs))
        no_of_trainbatch = int(len(self.train_dataset.files)/ cfg.batch_len)
        no_of_validbatch = int(len(self.valid_dataset.files)/ cfg.batch_len)
        print('batch size--> %d, seq len %d, no of batches--> %d' % (cfg.batch_len,cfg.seq_len, no_of_trainbatch))
        f.write('batch size--> %d, seq len %d, no of batches--> %d \n' % (cfg.batch_len,cfg.seq_len, no_of_trainbatch))

        min_batch_loss = 0.0
        min_valid_loss = 0.0

        try:
            ################ epoch loop ###################
            epoch_loss = {'train': [], 'validation': []}
            for epoch in range(n_epochs):
                train_loss = []
                start_time = time()
                ####################### training #######################
                self.train_dataset.loadfiles( self.dataPath, self.trainset)
                self.model.train()
                while (len(self.train_dataset.files) > 0):
                    self.train_dataset.prepareBatchOfMotion(10)
                    inputs = self.train_dataset.input
                    outputs = self.train_dataset.target
                    ##################### divide the data into chunk of seq len
                    chunk_in = list(torch.split(inputs,cfg.seq_len,dim=1))
                    chunk_out = list(torch.split(outputs, cfg.seq_len, dim=1))
                    if (len(chunk_in) == 0):
                        continue
                    print('chunk list size',len(chunk_in))
                    # initialize hidden and cell state  at each new batch
                    hidden = torch.zeros(cfg.n_layers * 2, chunk_in[0].shape[0], cfg.hid_dim, dtype=torch.double).cuda()
                    cell = torch.zeros(cfg.n_layers * 2, chunk_in[0].shape[0], cfg.hid_dim, dtype=torch.double).cuda()

                    for c_in,c_out in zip(chunk_in,chunk_out):
                        optimizer.zero_grad()
                        c_pred,hidden,cell = self.model(c_in,hidden,cell)
                        # hidden = hidden.detach()
                        # cell = cell.detach()
                        loss = self._loss_impl(c_pred,c_out)
                        loss.backward(retain_graph=True)
                        nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                        optimizer.step()
                        loss.detach()
                        train_loss.append(loss.item())

                train_loss = torch.mean(torch.FloatTensor(train_loss))
                epoch_loss['train'].append(train_loss)
                # we save the model after each epoch : epoch_{}.pth.tar
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'epoch_loss': train_loss
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
                self.valid_dataset.loadfiles( self.dataPath, self.testset)
                valid_loss = []
                while (len(self.valid_dataset.files) > 0):
                    self.valid_dataset.prepareBatchOfMotion(10)
                    inputs = self.valid_dataset.input
                    outputs = self.valid_dataset.target
                    # initialize hidden and cell state  at each new batch
                    hidden = torch.zeros(cfg.n_layers * 2, cfg.batch_len, cfg.hid_dim, dtype=torch.double).cuda()
                    cell = torch.zeros(cfg.n_layers * 2, cfg.batch_len, cfg.hid_dim, dtype=torch.double).cuda()

                    # divide the data into chunk of seq len
                    chunk_in = list(torch.split(inputs, cfg.seq_len, dim=1))
                    chunk_out = list(torch.split(outputs, cfg.seq_len, dim=1))
                    if (len(chunk_in) == 0):
                        continue
                    for c_in, c_out in zip(chunk_in, chunk_out):

                        c_pred,hidden,cell = self.model(c_in,hidden,cell)
                        hidden = hidden.detach()
                        cell = cell.detach()
                        loss = self._loss_impl(c_pred,c_out).detach()
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
        return torch.mean((torch.norm(L1, 2, 2)))



if __name__ == '__main__':
    trainingEngine = TrainingEngine()
    trainingEngine.train(n_epochs=50)