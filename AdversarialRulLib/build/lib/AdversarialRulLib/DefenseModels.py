import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, InputSize = 1920, OutputSize = 100, HidenForward1 = 1200, HidenForward2 = 600):
        super(Encoder, self).__init__()
        self.Linear1 = nn.Linear(InputSize, HidenForward1)
        self.Relu1 = nn.ReLU()
        self.Linear2 = nn.Linear(HidenForward1, HidenForward2)
        self.Relu2 = nn.ReLU()
        self.Linear3 = nn.Linear(HidenForward2, OutputSize)
        self.Relu3 = nn.ReLU()
    
    def forward(self, x):
        out = self.Linear1(x)
        out = self.Relu1(out)
        out = self.Linear2(out)
        out = self.Relu2(out)
        out = self.Linear3(out)
        out = self.Relu3(out)
        return out

class Decoder(nn.Module):
    def __init__(self, InputSize = 100, OutputSize = 1920, HidenForward1 = 600, HidenForward2 = 1200):
        super(Decoder, self).__init__()
        self.Linear1 = nn.Linear(InputSize, HidenForward1)
        self.Relu1 = nn.ReLU()
        self.Linear2 = nn.Linear(HidenForward1, HidenForward2)
        self.Relu2 = nn.ReLU()
        self.Linear3 = nn.Linear(HidenForward2, OutputSize)
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.Linear1(x)
        out = self.Relu1(out)
        out = self.Linear2(out)
        out = self.Relu2(out)
        out = self.Linear3(out)
        out = self.Sigmoid(out)
        return out
    
class Autoencoder(nn.Module):
    def __init__(self, InputSize = 1920, HiddenSize=100, device = "cpu"):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(InputSize = InputSize, OutputSize = HiddenSize).to(device)
        self.decoder = Decoder(InputSize = HiddenSize, OutputSize = InputSize).to(device)
    
    def forward(self, x):
        BatchSize, Window, FeatureSize = x.size()
        
        FlatX = x.view(BatchSize, Window * FeatureSize)
        Encoded = self.encoder(FlatX.float())
        Decoded = self.decoder(Encoded)
        
        NewX = Decoded.view(BatchSize, Window, FeatureSize)
        
        return NewX
    
class AutoEncoderDefenseModel(nn.Module):
    def __init__(self, AutoEncoder, Model, Device = "cpu"):
        super(AutoEncoderDefenseModel, self).__init__()
        self.Model = Model
        self.Encoder = AutoEncoder
        self.Device = Device
        for param in self.Encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.Encoder(x)
        out = self.Model(out)
        return out

class RandomizeDefenseModel(torch.nn.Module):
    def __init__(self, ExistingModel, Coefficient = 0.1, Device = "cpu"):
        super(RandomizeDefenseModel, self).__init__()
        self.ExistingModel = ExistingModel
        self.Coefficient = Coefficient
        self.Device = Device

    def forward(self, x): 
        Noise = torch.rand(x.size(), device=self.Device) * self.Coefficient*2 - self.Coefficient
        NoisyX = x + Noise
        NoisyX = torch.clamp(NoisyX, min=0.0, max=1.0)
        out = self.ExistingModel(NoisyX)
        return out
    
class RoundDefenseModel(torch.nn.Module):
    def __init__(self, ExistingModel, Coefficient = 0.1, Device = "cpu"):
        super(RoundDefenseModel, self).__init__()
        self.ExistingModel = ExistingModel
        self.Coefficient = Coefficient
        self.Device = Device

    def forward(self, x): 
        RoundX = torch.round(x * (1/self.Coefficient)) / (1/self.Coefficient)
        out = self.ExistingModel(RoundX)
        return out
    
class EnsembleRoundDefenseModel(torch.nn.Module):
    def __init__(self, ExistingModel, MaxCoefficient = 1, Device  = "cpu"):
        super(EnsembleRoundDefenseModel, self).__init__()
        self.ExistingModel = ExistingModel
        self.MaxCoefficient = MaxCoefficient
        self.Device = Device

    def forward(self, x): 
        Coefficient = 1
        Average = torch.zeros_like(self.ExistingModel(x))
        for i in range(1,self.MaxCoefficient+1):
            RoundX = torch.round(x * Coefficient) / Coefficient
            TemporaryOut = self.ExistingModel(RoundX)
            Average += TemporaryOut
            Coefficient *= 10
                
        Average /= self.MaxCoefficient
        return Average
    
class GRUDetectionModel(nn.Module):
    def __init__(self, InputSize = 24, HiddenSize = 10, NbLayers = 1):
        super(GRUDetectionModel, self).__init__()
        self.HiddenSize = HiddenSize
        self.NbLayers = NbLayers
        self.Gru = nn.GRU(InputSize, HiddenSize, NbLayers, batch_first=True)
        self.Linear1 = nn.Linear(HiddenSize, 1)
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.NbLayers, x.size(0), self.HiddenSize).to(x.device)
        out, _ = self.Gru(x, h0)
        out = self.Linear1(out[:, -1, :])
        out = self.Sigmoid(out)
        return out.squeeze()
    
class GRUStudentModel(nn.Module):
    def __init__(self, InputSize = 24, HiddenSize = 10, NbLayers = 1, HiddenForward = 12):
        super().__init__()
        self.InputSize = InputSize
        self.HiddenSize = HiddenSize
        self.NbLayers = NbLayers
        self.Gru = nn.GRU(input_size=InputSize, hidden_size=self.HiddenSize, batch_first=True, num_layers=self.NbLayers)
        self.Linear1 = nn.Linear(in_features=self.HiddenSize, out_features=HiddenForward)
        self.Relu1 = nn.ReLU()
        self.Linear2 = nn.Linear(in_features=HiddenForward, out_features=1)
        
    def forward(self, x):
        BatchSize = x.shape[0]
        Device = x.device
        
        h0 = torch.zeros(self.NbLayers, BatchSize, self.HiddenSize).requires_grad_().to(Device)
        
        _, hn = self.Gru(x, h0)
        out = self.Linear1(hn[0])
        out = self.Relu1(out)
        out = self.Linear2(out).flatten()
        return out
