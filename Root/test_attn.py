from Attention import RawDataset
from Attention import Encoder
from Attention import Decoder
import torch
import Config as cfg
import matplotlib.pyplot as plt
import numpy as np
import myUtil
#import quaternion
import itertools

class TestEngine:
    def __init__(self):

        self.datapath = '/data/Guha/GR/synthetic60FPS/'
        self.dataset = RawDataset()
        #self.modelPath = '/data/Guha/GR/model/18/'
        self.encoder = Encoder(input_dim=24,enc_units=256).cuda()
        self.decoder = Decoder(output_dim=60, dec_units=256, enc_units=256).cuda()
        baseModelPath = '/data/Guha/GR/model/attn_greedy/epoch_6.pth.tar'
        self.base = '/data/Guha/GR/Output/TestSet/attn/'

        with open(baseModelPath, 'rb') as tar:
            checkpoint = torch.load(tar)
            self.encoder.load_state_dict(checkpoint['encoder_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_dict'])

    def test(self):
        try:
            dset = ['AMASS_ACCAD', 'AMASS_BioMotion', 'AMASS_CMU_Kitchen', 'CMU', 'HEva', 'H36']

            ####################### Validation #######################
            self.dataset.loadfiles(self.datapath, ['H36'])
            valid_loss = []
            for file in self.dataset.files:
                # Pick a random sequence
                # file = '/data/Guha/GR/Dataset/DFKI/walking_1.npz'
                # file = '/data/Guha/GR/DIPIMUandOthers/DIP_IMU_and_Others/DIP_IMU/s_01/01.pkl'

                ################# test on synthetic data #########################
                chunk_in, chunk_target = self.dataset.readfile(file,cfg.seq_len)

                ################# test on DIP_IMU data #########################
                #input_batches, output_batches = self.dataset.readDIPfile(file,cfg.seq_len)

                ################# test on DFKI data #########################
                #input_batches = self.dataset.readDFKIfile(file, cfg.seq_len)

                self.encoder.eval()
                self.decoder.eval()
                if (len(chunk_in) == 0):
                    continue
                print('chunk list size', len(chunk_in))
                # pass all the chunks through encoder and accumulate c_out and c_hidden in a list
                enc_output = []
                enc_hidden = []
                for c_in in chunk_in:
                    # chunk_in: (batch_sz:10, seq_len: 200, in_dim: 20)
                    # chunk_out: (batch_sz:10, seq_len: 200, out_dim: 60)
                    c_in = c_in.unsqueeze(0)
                    c_enc_out, c_enc_hidden = self.encoder(c_in)
                    enc_output.append(c_enc_out)
                    enc_hidden.append(c_enc_hidden)

                # decoder input for the first timestep
                batch_sz = 1
                tpose = np.array([[1, 0, 0, 0] * 15] * batch_sz)
                dec_input = torch.FloatTensor(tpose.reshape(batch_sz, 1, 60)).cuda()

                # ########### for start with Ipose ###################
                # SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
                # ipose = myUtil.Ipose.reshape(-1, 24, 3)[:, SMPL_MAJOR_JOINTS, :]
                # qs = quaternion.from_rotation_vector(ipose)
                # qs = quaternion.as_float_array(qs)
                # dec_input = torch.FloatTensor(qs.reshape(1, 1, 60)).cuda()
                dec_input = chunk_target[0][0, :].reshape(1,1,60)

                # pass all chunks to the decoder and predict for each timestep for all chunks sequentially
                predictions = []
                for c_enc_out, c_enc_hidden, c_target in zip(enc_output, enc_hidden, chunk_target):
                    dec_hidden = c_enc_hidden
                    loss = 0.0
                    for t in range(c_target.shape[0]):
                        pred_t, dec_hidden = self.decoder(dec_input, dec_hidden, c_enc_out)
                        dec_input = pred_t.unsqueeze(1)
                        #loss += self._loss_impl(c_target[t], pred_t)
                        predictions.append(pred_t.detach().cpu().numpy())


                target = torch.cat(chunk_target).detach().cpu().numpy()
                predictions = np.asarray(predictions).reshape(-1,15,4)
                norms = np.linalg.norm(predictions, axis=2)
                predictions = np.asarray(
                    [predictions[k, j, :] / norms[0, 0] for k, j in itertools.product(range(predictions.shape[0]), range(15))])
                np.savez_compressed(self.base + file.split('/')[-1], target=target, predictions=predictions)
                #np.savez_compressed(self.base + file.split('/')[-1], predictions=pred)
                #break

        except KeyboardInterrupt:
            print('Testing aborted.')

    def _loss_impl(self, predicted, expected):
        L1 = predicted - expected
        return torch.mean((torch.norm(L1, 2, 1)))



if __name__ == '__main__':
    trainingEngine = TestEngine()
    trainingEngine.test()