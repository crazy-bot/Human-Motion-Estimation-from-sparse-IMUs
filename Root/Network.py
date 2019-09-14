import torch
import torch.nn as nn
import Config as cfg


class BiRNN(nn.Module):
    def __init__(self):
        super(BiRNN,self).__init__()

        self.input_dim = cfg.input_dim
        self.hid_dim = cfg.hid_dim
        self.n_layers = cfg.n_layers
        self.dropout = cfg.dropout

        self.relu = nn.ReLU()
        self.pre_fc = nn.Linear(cfg.input_dim , cfg.hid_dim)
        self.lstm = nn.LSTM(cfg.hid_dim, cfg.hid_dim, cfg.n_layers, batch_first=True, dropout=cfg.dropout,bidirectional=True)
        self.post_fc = nn.Linear(cfg.hid_dim*2,cfg.output_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, X):
        # src = [ batch size, seq len, input dim]
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        input_dim = X.shape[2]
        #X = torch.Tensor(src)
        X = X.view(-1,input_dim)
        X = self.pre_fc(X)
        X = self.relu(X)
        X = X.view(batch_size,seq_len, -1)
        lstm_out, (hidden, cell) = self.lstm(X)

        """lstm_out : [batch size, src sent len, hid dim * n directions]
        hidden : [n layers * n directions, batch size, hid dim]
        cell : [n layers * n directions,batch size, hid dim]
        lstm_out are always from the top hidden layer """

        fc_out = self.post_fc(lstm_out)

        return fc_out

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM,self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout)
        self.pre_fc = nn.Sequential(nn.Linear(cfg.input_dim, cfg.hid_dim),self.relu)
        self.lstm = nn.LSTM(cfg.hid_dim, cfg.hid_dim, cfg.n_layers, batch_first=True, dropout=cfg.dropout, bidirectional=True)
        self.post_fc = nn.Linear(cfg.hid_dim * 2, cfg.output_dim)


    def forward(self, X, h_0, c_0):
        X = self.pre_fc(X)
        """lstm_out : [batch size, src sent len, hid dim * n directions]
                hidden : [n layers * n directions, batch size, hid dim]
                cell : [n layers * n directions,batch size, hid dim]
                lstm_out are always from the top hidden layer """

        lstm_out, (hidden, cell) = self.lstm(X,(h_0,c_0))
        hidden,cell = (hidden, cell)
        fc_out = self.post_fc(lstm_out)
        return fc_out,hidden,cell

class ForwardKinematic(nn.Module):
    def __init__(self):
        super(ForwardKinematic,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(60,120), nn.ReLU(),nn.Linear(120,180),nn.Dropout(0.3),
            nn.ReLU(),nn.Linear(180,60),nn.Dropout(0.3),nn.ReLU(),nn.Linear(60,20)
        )
    def forward(self, input):
        out = self.net(input)
        return out

class InverseKinematic(nn.Module):
    def __init__(self):
        super(InverseKinematic,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(20,60), nn.ReLU(),nn.Linear(60,180),nn.ReLU(),nn.Linear(180,60)
        )
    def forward(self, input):
        out = self.net(input)
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super(Encoder).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout


        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, batch_first=True, dropout=dropout,bidirectional=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [ batch size, seq len, input dim]

        outputs, (hidden, cell) = self.rnn(src)

        # outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Decoder).__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, batch_first=True, dropout=dropout,bidirectional=True)

        self.fc = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [batch size , 1, output_dim ]
        output, (hidden, cell) = self.rnn(input, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell