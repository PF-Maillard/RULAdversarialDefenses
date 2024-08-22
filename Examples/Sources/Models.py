import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, FeatureSize = 24, NbRNNLayers = 1, HiddenRNN = 12, HiddenForward = 12, HiddenForward2 = 12):
        super().__init__()
        self.FeatureSize = FeatureSize
        self.HiddenRNN = HiddenRNN
        self.NbLayers = NbRNNLayers
        self.Rnn = nn.RNN(input_size=FeatureSize, hidden_size=self.HiddenRNN, batch_first=True, num_layers=self.NbLayers)
        self.Linear1 = nn.Linear(in_features=self.HiddenRNN, out_features=HiddenForward)
        self.Relu1 = nn.ReLU()
        self.Linear2 = nn.Linear(in_features=HiddenForward, out_features=HiddenForward2)
        self.Relu2 = nn.ReLU()
        self.Linear3 = nn.Linear(in_features=HiddenForward2, out_features=1)
        
    def forward(self, x):
        BatchSize = x.shape[0]
        Device = x.device
        
        h0 = torch.zeros(self.NbLayers, BatchSize, self.HiddenRNN).requires_grad_().to(Device)
        
        self.Rnn.flatten_parameters()
        _, hn = self.Rnn(x, h0)
        out = self.Linear1(hn[0])
        out = self.Relu1(out)
        out = self.Linear2(out)
        out = self.Relu2(out)
        out = self.Linear3(out).flatten()
        return out

class GRUModel(nn.Module):
    def __init__(self, FeatureSize = 24, NbGRULayers = 1, HiddenGRU = 12, HiddenForward = 12, HiddenForward2 = 12):
        super().__init__()
        self.FeatureSize = FeatureSize
        self.HiddenGRU = HiddenGRU
        self.NbLayers = NbGRULayers
        self.Gru = nn.GRU(input_size=FeatureSize, hidden_size=self.HiddenGRU, batch_first=True, num_layers=self.NbLayers)
        self.Linear1 = nn.Linear(in_features=self.HiddenGRU, out_features=HiddenForward)
        self.Relu1 = nn.ReLU()
        self.Linear2 = nn.Linear(in_features=HiddenForward, out_features=HiddenForward2)
        self.Relu2 = nn.ReLU()
        self.Linear3 = nn.Linear(in_features=HiddenForward2, out_features=1)
        
    def forward(self, x):
        
        BatchSize = x.shape[0]
        Device = x.device
        
        h0 = torch.zeros(self.NbLayers, BatchSize, self.HiddenGRU).requires_grad_().to(Device)
        
        self.Gru.flatten_parameters()
        _, hn = self.Gru(x, h0)
        out = self.Linear1(hn[0])
        out = self.Relu1(out)
        out = self.Linear2(out)
        out = self.Relu2(out)
        out = self.Linear3(out).flatten()
        return out

class LSTMModel(nn.Module):
    
    def __init__(self, FeatureSize = 24, NbLSTMLayers = 1, HiddenLSTM = 12, HiddenForward = 12, HiddenForward2 = 12):
        super().__init__()
        self.FeatureSize = FeatureSize
        self.HiddenLSTM = HiddenLSTM
        self.NbLayers = NbLSTMLayers
        self.Lstm = nn.LSTM(input_size = FeatureSize, hidden_size = self.HiddenLSTM, batch_first = True, num_layers = self.NbLayers)
        self.Linear1 = nn.Linear(in_features=self.HiddenLSTM, out_features=HiddenForward)
        self.Relu1 = nn.ReLU()
        self.Linear2 = nn.Linear(in_features=HiddenForward, out_features=HiddenForward2)
        self.Relu2 = nn.ReLU()
        self.Linear3 = nn.Linear(in_features=HiddenForward2, out_features=1)
        
    def forward(self, x):
        BatchSize = x.shape[0]
        Device = x.device
        
        h0 = torch.zeros(self.NbLayers, BatchSize, self.HiddenLSTM).requires_grad_().to(Device)
        c0 = torch.zeros(self.NbLayers, BatchSize, self.HiddenLSTM).requires_grad_().to(Device)
        
        self.Lstm.flatten_parameters()
        _, (hn, _) = self.Lstm(x, (h0, c0))
        out = self.Linear1(hn[0])
        out = self.Relu1(out)
        out = self.Linear2(out)
        out = self.Relu2(out)
        out = self.Linear3(out).flatten()
        
        return out
    