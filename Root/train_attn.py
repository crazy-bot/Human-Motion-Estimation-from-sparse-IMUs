from Attention import RawDataset
from Attention import Encoder
from Attention import Decoder
import torch.optim as optim
from time import time
import torch
import Config as cfg
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np


class TrainingEngine:
    def __init__(self):
        self.trainset = ['H36_S1_S9','AMASS_HDM05']
        self.testset = ['H36_S11']
        self.datapath = '/data/Guha/GR/synthetic60FPS/'
        #self.datapath = '/ds2/synthetic60FPS/synthetic60FPS/'
        self.dataset = RawDataset()
        self.validset = RawDataset()
        self.use_cuda = True
        self.modelPath = '/data/Guha/GR/model/attn/'
        #self.modelPath = '/b_test/suparna/20/'
        self.encoder = Encoder(input_dim=24, enc_units=256).cuda()
        self.decoder = Decoder(output_dim=60, dec_units=256, enc_units=256).cuda()
        # baseModelPath = '/data/Guha/GR/model/19/epoch_3.pth.tar'
        #
        # with open(baseModelPath, 'rb') as tar:
        #     checkpoint = torch.load(tar)
        #     self.encoder.load_state_dict(checkpoint['encoder_dict'])
        #     self.decoder.load_state_dict(checkpoint['decoder_dict'])

    def train(self, n_epochs):
        f = open(self.modelPath + 'model_details', 'w')
        f.write(str(self.encoder))
        f.write('\n')
        f.write(str(self.decoder))
        f.write('\n')
        group_sz = 10
        np.random.seed(1234)
        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.001)

        min_valid_loss = 0.0
        print('Training for {} epochs'.format(n_epochs))
        self.dataset.loadfiles(self.datapath,self.trainset)
        print('total no of files {}'.format(len(self.dataset.files)))
        f.write('total no of files {} \n'.format(len(self.dataset.files)))

        try:
            ################ epoch loop ###################
            epoch_loss = {'train': [], 'validation': []}
            for epoch in range( n_epochs):
                start_time = time()
                ####################### training #######################
                self.encoder.train()
                self.decoder.train()
                train_loss = []
                self.dataset.loadfiles(self.datapath, self.trainset)
                while (len(self.dataset.files) > 0):
                    self.dataset.prepareBatchOfMotion(group_sz)
                    inputs = self.dataset.input
                    outputs = self.dataset.target
                    # divide the data into chunk of seq len
                    chunk_in = list(torch.split(inputs, cfg.seq_len, dim=1))
                    chunk_target = list(torch.split(outputs, cfg.seq_len, dim=1))

                    if (len(chunk_in) == 0):
                        continue
                    print('chunk list size',len(chunk_in))
                    # pass all the chunks through encoder and accumulate c_out and c_hidden in a list
                    enc_output = []
                    enc_hidden = []
                    for c_in in chunk_in:
                        # chunk_in: (batch_sz:10, seq_len: 200, in_dim: 20)
                        #chunk_out: (batch_sz:10, seq_len: 200, out_dim: 60)
                        c_enc_out,c_enc_hidden = self.encoder(c_in)
                        enc_output.append(c_enc_out)
                        enc_hidden.append(c_enc_hidden)

                    # decoder input for the first timestep
                    batch_sz = chunk_in[0].shape[0]
                    tpose = np.array([[1, 0, 0, 0] * 15] * batch_sz)
                    dec_input = torch.FloatTensor(tpose.reshape(batch_sz, 1, 60)).cuda()

                    # pass all chunks to the decoder and predict for each timestep for all chunks sequentially
                    #predictions = []
                    for c_enc_out, c_enc_hidden , c_target in zip(enc_output,enc_hidden,chunk_target):
                        dec_hidden = c_enc_hidden
                        loss = 0.0
                        for t in range(c_target.shape[1]):
                            pred_t, dec_hidden = self.decoder(dec_input, dec_hidden, c_enc_out)
                            dec_input = pred_t.detach().unsqueeze(1)
                            loss += self._loss_impl(c_target[:,t], pred_t)
                            #predictions.append(pred_t.detach())

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.item() / (t+1) )

                train_loss = torch.mean(torch.FloatTensor(train_loss))
                epoch_loss['train'].append(train_loss)
                # we save the model after each epoch : epoch_{}.pth.tar
                state = {
                    'epoch': epoch + 1,
                    'encoder_dict': self.encoder.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'epoch_loss': train_loss
                }
                torch.save(state, self.modelPath + 'epoch_{}.pth.tar'.format(epoch + 1))

                debug_string = 'epoch No {}, epoch loss {}, Time taken {} \n'.format(
                    epoch + 1, train_loss, start_time - time()
                )
                print(debug_string)
                f.write(debug_string)
                f.write('\n')
                ####################### Validation #######################
                self.validset.loadfiles(self.datapath, self.testset)
                valid_loss = []
                while (len(self.validset.files) > 0):
                    self.validset.prepareBatchOfMotion(2)
                    inputs = self.validset.input
                    outputs = self.validset.target
                    # divide the data into chunk of seq len
                    chunk_in = list(torch.split(inputs, cfg.seq_len, dim=1))
                    chunk_target = list(torch.split(outputs, cfg.seq_len, dim=1))
                    if (len(chunk_in) == 0):
                        continue
                    # pass all the chunks through encoder and accumulate c_out and c_hidden in a list
                    enc_output = []
                    enc_hidden = []
                    for c_in in chunk_in:
                        # chunk_in: (batch_sz:10, seq_len: 200, in_dim: 20)
                        # chunk_out: (batch_sz:10, seq_len: 200, out_dim: 60)
                        c_enc_out, c_enc_hidden = self.encoder(c_in)
                        enc_output.append(c_enc_out)
                        enc_hidden.append(c_enc_hidden)

                    # decoder input for the first timestep
                    batch_sz = chunk_in[0].shape[0]
                    tpose = np.array([[1, 0, 0, 0] * 15] * batch_sz)
                    dec_input = torch.FloatTensor(tpose.reshape(batch_sz, 1, 60)).cuda()

                    # pass all chunks to the decoder and predict for each timestep for all chunks sequentially
                    # predictions = []
                    for c_enc_out, c_enc_hidden, c_target in zip(enc_output, enc_hidden, chunk_target):
                        dec_hidden = c_enc_hidden
                        loss = 0.0
                        for t in range(c_target.shape[1]):
                            pred_t, dec_hidden = self.decoder(dec_input, dec_hidden, c_enc_out)
                            ############### use of teacher forcing in greedy method #############
                            # Return random floats in the half-open interval [0.0, 1.0)
                            rand = np.random.random()
                            # teacher forcing time
                            if (rand > 1 - self.tf_rate):
                                dec_input = c_target[:, t].unsqueeze(1)
                            # use own prediction
                            else:
                                dec_input = pred_t.detach().unsqueeze(1)
                            loss += self._loss_impl(c_target[:, t], pred_t)
                            # predictions.append(pred_t.detach())
                        valid_loss.append(loss.item() / (t + 1))

                valid_loss = torch.mean(torch.FloatTensor(valid_loss))
                epoch_loss['validation'].append(valid_loss)
                # we save the model if current validation loss is less than prev : validation.pth.tar
                if (min_valid_loss == 0 or valid_loss < min_valid_loss):
                    min_valid_loss = valid_loss
                    state = {
                        'epoch': epoch + 1,
                        'encoder_dict': self.encoder.state_dict(),
                        'decoder_dict': self.decoder.state_dict(),
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
            state = {
                'epoch': epoch + 1,
                'encoder_dict': self.encoder.state_dict(),
                'decoder_dict': self.decoder.state_dict(),

            }
            torch.save(state, self.modelPath + 'error.pth.tar')
            print('Training aborted.')

    def _loss_impl(self, predicted, expected):
        L1 = predicted - expected
        return torch.mean((torch.norm(L1, 2, 1)))


if __name__ == '__main__':
    trainingEngine = TrainingEngine()
    trainingEngine.train(n_epochs=50)