import torch
import torch.nn as nn
import torch.optim as optim
import Config as cfg
import numpy as np
from time import time
import random
from IMUDataset import IMUDataset

class BiRNN(nn.Module):
    def __init__(self):
        super(BiRNN,self).__init__()

        self.input_dim = cfg.input_dim
        self.hid_dim = 256
        self.n_layers = cfg.n_layers
        self.dropout = cfg.dropout

        self.relu = nn.ReLU()
        self.pre_fc = nn.Linear(cfg.input_dim , 256)
        self.lstm = nn.LSTM(256, 256, cfg.n_layers, batch_first=True, dropout=cfg.dropout,bidirectional=True)
        self.post_fc = nn.Linear(256*2,cfg.output_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, X):
        # src = [ batch size, seq len, input dim]
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        input_dim = X.shape[2]

        X = X.view(-1,input_dim)
        X = self.pre_fc(X)
        X = self.relu(X)
        X = X.view(batch_size,seq_len, -1)
        lstm_out, (_, _) = self.lstm(X)

        """lstm_out : [batch size, src sent len, hid dim * n directions]
        hidden : [n layers * n directions, batch size, hid dim]
        cell : [n layers * n directions,batch size, hid dim]
        lstm_out are always from the top hidden layer """

        fc_out = self.post_fc(lstm_out)
        return fc_out

class TrainingEngine:
    def __init__(self):
        self.datapath = '/data/Guha/GR/Dataset/'
        self.modelPath = '/data/Guha/GR/model/H36_DIP/'
        self.model = BiRNN().cuda()
        self.trainset = ['H36','DIP_IMU/train']
        self.testset = ['DIP_IMU/validation']
        # baseModelPath = '/data/Guha/GR/model/13/epoch_5.pth.tar'
        #
        # with open(baseModelPath, 'rb') as tar:
        #     checkpoint = torch.load(tar)
        #     model_weights = checkpoint['state_dict']
        # self.model.load_state_dict(model_weights)

    def _loss_impl(self,predicted, expected):
        L1 = predicted - expected
        return torch.mean((torch.norm(L1, 2, 2)))

    def train(self,n_epochs):
        f = open(self.modelPath+'model_details','w')
        f.write(str(self.model))
        f.write('\n')

        np.random.seed(1234)
        lr = 0.001
        gradient_clip = 0.1
        optimizer = optim.Adam(self.model.parameters(),lr=lr)

        print('Training for %d epochs' % (n_epochs))

        print('batch size--> {}, Seq len--> {}'.format(cfg.batch_len, cfg.seq_len))
        f.write('batch size--> {}, Seq len--> {} \n'.format(cfg.batch_len, cfg.seq_len))
        epoch_loss = {'train': [], 'validation': []}
        self.dataset = IMUDataset(self.datapath, self.trainset)
        min_valid_loss = 0.0
        for epoch in range(0, n_epochs):
            train_loss = []
            start_time = time()
            self.dataset.loadfiles(self.datapath,self.trainset)
            ####################### training #######################
            while (len(self.dataset.files) > 0):
                # Pick a random chunk from each sequence
                self.dataset.createbatch_no_replacement()
                inputs = torch.FloatTensor(self.dataset.input).cuda()
                outputs = torch.FloatTensor(self.dataset.target).cuda()
                chunk_in = list(torch.split(inputs, cfg.seq_len))[:-1]
                chunk_out = list(torch.split(outputs, cfg.seq_len))[:-1]
                if (len(chunk_in) == 0):
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

                train_loss.append(loss.item())

            train_loss = torch.mean(torch.FloatTensor(train_loss))
            # we save the model after each epoch : epoch_{}.pth.tar
            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'epoch_loss': train_loss
            }
            torch.save(state, self.modelPath + 'epoch_{}.pth.tar'.format(epoch + 1))
            # logging to track
            debug_string = 'epoch No {}, training loss {} , Time taken {} \n'.format(
                epoch + 1, train_loss,  start_time - time()
            )
            print(debug_string)
            f.write(debug_string)
            f.write('\n')
            epoch_loss['train'].append(train_loss)
            ####################### Validation #######################
            valid_loss = []
            self.model.eval()
            self.dataset.loadfiles(self.datapath, self.testset)
            for file in self.dataset.files:
                self.dataset.readfile(file)
                input = torch.FloatTensor(self.dataset.input).unsqueeze(0).cuda()
                target = torch.FloatTensor(self.dataset.target).unsqueeze(0).cuda()

                output = self.model(input)
                loss = self._loss_impl(output, target)
                valid_loss.append(loss)
            valid_loss = torch.mean(torch.FloatTensor(valid_loss))
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
            debug_string = 'epoch No {}, valid loss {} ,Time taken {} \n'.format(
                epoch + 1, valid_loss, start_time - time()
            )
            print(debug_string)
            f.write(debug_string)
            f.write('\n')
            epoch_loss['validation'].append(valid_loss)

        f.write(str(epoch_loss))
        f.close()
        self.plotGraph(epoch_loss, self.modelPath)


    def plotGraph(epoch_loss,basepath):
        import  matplotlib.pyplot as plt
        fig = plt.figure(1)
        trainloss = epoch_loss['train']
        validloss = epoch_loss['validation']

        plt.plot(np.arange(trainloss), trainloss, 'r--', label='training loss')
        plt.plot(np.arange(validloss), validloss, 'g--', label='validation loss')
        plt.legend()
        plt.savefig(basepath+'.png')
        plt.show()

if __name__ == "__main__":
    trainingEngine = TrainingEngine()
    trainingEngine.train(n_epochs=50)