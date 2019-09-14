import numpy as np
from Network import BiRNN,CorrectPose
import torch.optim as optim
from IMUDataset import IMUDataset
from time import time
import torch
import Config as cfg
import torch.nn as nn
from eval_model import TestEngine


class TrainingEngine:
    def __init__(self):
        self.train_dataset = IMUDataset(cfg.traindata_path)
        self.valid_dataset = IMUDataset(cfg.validdata_path)
        self.use_cuda =True
        self.modelPath = '/data/Guha/GR/model/11/'
        self.poseEstimator = BiRNN()
        self.mseloss = nn.MSELoss()
        baseModelPath = '/data/Guha/GR/model/9/validation.pth.tar'

        with open(baseModelPath, 'rb') as tar:
            checkpoint = torch.load(tar)
            model_weights = checkpoint['state_dict']

        self.poseEstimator.load_state_dict(model_weights)
        self.poseCorrector = CorrectPose(cfg.input_dim+cfg.output_dim,cfg.output_dim)
        self.poseEstimator.cuda().eval()
        self.poseCorrector.cuda()



    def train(self,n_epochs):

        f = open(self.modelPath+'model_details','w')
        f.write(str(self.poseCorrector))
        f.write('\n')

        np.random.seed(1234)
        lr = 0.001
        optimizer = optim.Adam(self.poseCorrector.parameters(),lr=lr)

        print('Training for %d epochs' % (n_epochs))

        min_batch_loss = 0.0
        min_valid_loss = 0.0

        try:
            for epoch in range(n_epochs):
                epoch_loss = []
                start_time = time()
                self.poseCorrector.train()
                for tf in self.train_dataset.files:
                    self.train_dataset.readfile(tf)
                    input = torch.FloatTensor(self.train_dataset.input)
                    input = torch.unsqueeze(input, 0)
                    target = torch.FloatTensor(self.train_dataset.target)
                    if self.use_cuda:
                        input = input.cuda()
                        target = target.cuda()
                    # pose prediction
                    pose_prediction = self.poseEstimator(input).squeeze(0).detach()
                    #target = target.reshape_as(pose_prediction)
                    in_correct = torch.cat((input.squeeze(0),pose_prediction),1)
                    # pose correction
                    pose_correction = self.poseCorrector(in_correct)
                    pose_correction = pose_correction.reshape_as(target)
                    loss = self._loss_impl(pose_correction, target)
                    loss.backward()
                    optimizer.step()
                    loss.detach()
                    epoch_loss.append(loss.item())
                    if (min_batch_loss == 0 or loss < min_batch_loss):
                        min_batch_loss = loss
                        print ('file ----------> %{} training loss {} '.format(tf, loss.item()))
                        f.write('file ----------> %{} training loss {} \n'.format(tf, loss.item()))

                epoch_loss = torch.mean(torch.FloatTensor(epoch_loss))
                # we save the model after each epoch : epoch_{}.pth.tar
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.poseCorrector.state_dict(),
                    'epoch_loss': epoch_loss
                }
                torch.save(state, self.modelPath + 'epoch_{}.pth.tar'.format(epoch+1))

                ####################### Validation #######################
                valid_loss = []
                self.poseCorrector.eval()
                for vf in self.valid_dataset.files:
                    self.valid_dataset.readfile(vf)
                    input = torch.FloatTensor(self.valid_dataset.input)
                    input = torch.unsqueeze(input, 0)
                    target = torch.FloatTensor(self.valid_dataset.target)
                    if self.use_cuda:
                        input = input.cuda()
                        target = target.cuda()
                    # pose prediction
                    pose_prediction = self.poseEstimator(input).squeeze(0)
                    #target = target.reshape_as(pose_prediction)
                    in_correct = torch.cat((input.squeeze(0),pose_prediction),1)
                    # pose correction
                    pose_correction = self.poseCorrector(in_correct)
                    pose_correction = pose_correction.reshape_as(target)
                    loss = self._loss_impl(pose_correction, target)
                    loss.detach()
                    valid_loss.append(loss.item())

                valid_loss = torch.mean(torch.FloatTensor(valid_loss))
                # we save the model if current validation loss is less than prev : validation.pth.tar
                if (min_valid_loss == 0 or valid_loss < min_valid_loss):
                    min_valid_loss = valid_loss
                    state = {
                        'epoch': epoch + 1,
                        'state_dict': self.poseCorrector.state_dict(),
                        'validation_loss': valid_loss
                    }
                    torch.save(state, self.modelPath + 'validation.pth.tar')

                # logging to track
                print ('epoch No %d, epoch loss %d , validation loss %d, Time taken %d \n' % (
                epoch + 1, epoch_loss, valid_loss, start_time - time()))
                f.write('epoch No %d, epoch loss %d , validation loss %d, Time taken %d \n' % (
                epoch + 1, epoch_loss, valid_loss, start_time - time()))

            f.close()
        except KeyboardInterrupt:
            print('Training aborted.')

    def _loss_impl(self, predicted, expected):
        L1 = predicted - expected
        dist = torch.sum(torch.norm(L1, 2, 2))
        return  dist/ predicted.shape[0]



if __name__ == '__main__':
    trainingEngine = TrainingEngine()
    trainingEngine.train(n_epochs=30)