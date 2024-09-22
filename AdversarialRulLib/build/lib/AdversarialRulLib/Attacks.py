import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

from . import Utils as UtilsTool
from .DefenseModels import GRUStudentModel

def Fgsm(Model, Objective, X, y, Epsilon = 0.2, Device = "cpu"):
    Model.train()
    Adv = X.clone()
    Adv.requires_grad = True
    criterion = nn.MSELoss()
    
    output = Model(Adv)
    Target = torch.full_like(y, Objective)
    loss = criterion(output, Target)
    loss.backward()
    Grad = Adv.grad.data
    SX = Grad.sign().to(Device)
    Adv = Adv + Epsilon * SX
    Adv = torch.clamp(Adv, min=0.0, max=1.0)
    return Adv

def Bim(Model, Objective, X, y, Epsilon = 0.01, Epochs = 30, Device = "cpu"):
    Model.train()
    Adv = X.clone()
    criterion = nn.MSELoss()
    
    for i in range(Epochs):
        Adv.requires_grad = True
        output = Model(Adv)
        Target = torch.full_like(y, Objective)
        loss = criterion(output, Target)
        loss.backward()
        Grad = Adv.grad.data
        
        SX = Grad.sign().to(Device)
        with torch.no_grad():
            Adv = Adv + Epsilon * SX
            
    Adv = torch.clamp(Adv, min=0.0, max=1.0)
    return Adv

def L2(Model, Objective, X, y, LearningRate = 0.01, c = 1, Epochs = 100, Device = "cpu"):
    Model.train()
    Adv = X.clone()
    Adv.requires_grad = True
    Optimizer = optim.Adam([Adv], lr=LearningRate)
    MseLoss = nn.MSELoss()
    
    for i in range(Epochs):
        Optimizer.zero_grad()
        Output = Model(Adv)
        Target = torch.full_like(y, Objective)
        TargetLoss = MseLoss(Output, Target)
        DiffLoss = MseLoss(Adv, X)
        loss = c * TargetLoss +  DiffLoss
        
        loss.backward()
        Optimizer.step()
        with torch.no_grad():
            Adv.clamp_(0.0, 1.0)
    return Adv


class L0Loss(nn.Module):
    
    def tanh(self, x, c):
        return torch.tanh(x*c)
    
    def __init__(self, k, device):
        self.k=k
        self.device = device
        super(L0Loss, self).__init__()

    def forward(self, X, Z):
        Focus = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        Focus2D = Focus.unsqueeze(0).repeat(len(X[0]), 1)
        Focus2D = Focus2D.to(self.device)
        
        L2 = ((X - Z) ** 2)
        L2*= Focus2D
        Result = self.tanh(L2, self.k)
        Result2 = torch.sum(Result, dim=1)
        Result3 = self.tanh(Result2, self.k)

        loss = torch.sum(Result3)
        return loss

def Project(Adv, X, S, Loss):
    L2 = ((Adv - X) ** 2)
    Result = Loss.tanh(L2, Loss.k)
    Result2 = torch.sum(Result, dim=1)
    Result3 = Loss.tanh(Result2, Loss.k)
    
    ToProject = []
    for i in range(len(Result3[0])):
        if Result3[0][i] < S:
            ToProject.append(i)
    
    for Ins in range(len(Adv[0])):
        for i in ToProject:
            Adv[0][Ins][i] = X[0][Ins][i]
    return Adv

def L0(Model, Objective, X, y, LearningRate=0.01, c=0.01, Epochs = 1000, k= 2, p= 8, s= 0.1, Device= "cpu"):
    Model = Model.to(Device)
    Model.train()
    
    LAdversarial =[]
    for Instance in range(len(X)):
        Adv = X[Instance:Instance+1].clone()
        Adv.requires_grad = True
        Optimizer = optim.Adam([Adv], lr=LearningRate)
        MyLoss = L0Loss(k, Device)
        MseLoss = nn.MSELoss()
        
        MinLoss = 100000000
        Target = torch.full_like(y[Instance:Instance+1], Objective)
        SavedAdv = Adv.clone()
        
        for i in range(Epochs):
            token = 0
            
            with torch.no_grad():
                Adv.clamp_(0.0, 1.0)
                if random.randint(0, p) == 0:
                    Adv = Project(Adv, X[Instance:Instance+1], s, MyLoss)    
                    token = 1   
            
            Output = Model(Adv)
            
            TargetLoss = MseLoss(Output, Target)
            DiffLoss = MyLoss(Adv, X[Instance:Instance+1])
            
            loss = c * TargetLoss +  DiffLoss
            
            if token == 1 and loss.item() < MinLoss:
                MinLoss = loss.item()
                SavedAdv = Adv.clone()
                
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
        
        LAdversarial.append(SavedAdv) 
    FAdversarial = torch.cat(LAdversarial, dim=0)
    
    return FAdversarial

def TestAttacks(Model, X, y, AttacksParameters, Device = "cpu"):
    
    X, y = X.to(Device).to(torch.float32), y.to(Device).to(torch.float32)

    AdversarialDataFgsm = Fgsm(Model, AttacksParameters["FGSM"]["Objective"], X, y, Epsilon = AttacksParameters["FGSM"]["Epsilon"], Device = Device)
    AdversarialDataBim = Bim(Model, AttacksParameters["BIM"]["Objective"], X, y, Epsilon = AttacksParameters["BIM"]["Epsilon"], Epochs = AttacksParameters["BIM"]["Iterations"], Device = Device)
    AdversarialDataL2 = L2(Model, AttacksParameters["L2"]["Objective"], X, y, LearningRate = AttacksParameters["L2"]["LearningRate"], c = AttacksParameters["L2"]["c"], Epochs = AttacksParameters["L2"]["Iterations"], Device = Device)
    AdversarialDataL0 = L0(Model, AttacksParameters["L0"]["Objective"], X, y, LearningRate = AttacksParameters["L0"]["LearningRate"], c = AttacksParameters["L0"]["c"], Epochs = AttacksParameters["L0"]["Iterations"], k = AttacksParameters["L0"]["k"], p = AttacksParameters["L0"]["p"], s = AttacksParameters["L0"]["s"],Device = Device)
    
    Infos = UtilsTool.GetInfos(X, y, 0, AdversarialDataFgsm, Model)
    print("FGSM:", Infos)

    Infos = UtilsTool.GetInfos(X, y, 0, AdversarialDataBim, Model)
    print("BIM:", Infos)

    Infos = UtilsTool.GetInfos(X, y, 0, AdversarialDataL2, Model)
    print("L2:", Infos)
    
    Infos = UtilsTool.GetInfos(X, y, 0, AdversarialDataL0, Model)
    print("L0:", Infos)
    
    return AdversarialDataFgsm, AdversarialDataBim, AdversarialDataL2, AdversarialDataL0

class FakeDataset(Dataset):
    def __init__(self, NbSamples, Window=80, NbFeatures=24):
        self.NbSamples = NbSamples
        self.Window = Window
        self.NbFeatures = NbFeatures

    def __len__(self):
        return self.NbSamples

    def __getitem__(self, idx):
        Data = torch.rand(self.Window, self.NbFeatures)
        Target = torch.zeros(1)
        return Data, Target

def CreateSurrogateModel(Model, Student = None, Epochs = 30, LearningRate = 0.01, NbSamples = 100000, Window =80, NbFeatures = 24,  Device = "cpu", Verbose = 0):

    MyDataset = FakeDataset(NbSamples, Window =Window, NbFeatures = NbFeatures)
    MyDataloader = DataLoader(MyDataset, batch_size=64, shuffle=True)
    
    if  Student == None:
        if Verbose == 1:
            print("No Student model provided, default model used")
        Student = GRUStudentModel(MyDataset.NbFeatures, 20).to(Device)

    Criterion = nn.MSELoss()
    Optimizer = torch.optim.Adam(Student.parameters(), lr=LearningRate)

    Student.train()
    for i in tqdm(range(Epochs)):
        TrainLoss = 0
        for batch, (x,y) in enumerate(MyDataloader):
            x, _ = x.to(Device).to(torch.float32), y.to(Device).to(torch.float32)
            x_numpy = UtilsTool.Flatten_sequences(x.detach().cpu().numpy())
            TeacherPred = Model.predict(x_numpy)  
            Studentpred = Student(x.float())
            TeacherPred_torch = torch.tensor(TeacherPred).to(Device)
            Loss = Criterion(Studentpred.float(), TeacherPred_torch.float())
            TrainLoss += Loss.item()
            Optimizer.zero_grad()
            Loss.backward()
            Optimizer.step()
        
        if Verbose == 1 and (i+1) % 1 == 0:
            print(f'Epoch:{i+1}, Train Loss:{TrainLoss/len(MyDataloader)}')

    return Student

def CreateSurrogateModelSqueezing(Model, Student = None, Epochs = 30, LearningRate = 0.01, Device = "cpu", Verbose = 0):
    MyDataset = FakeDataset(1000000)
    MyDataloader = DataLoader(MyDataset, batch_size=2048, shuffle=True)
     
    if  Student == None:
        if Verbose == 1:
            print("No Student model provided, default model used")
        Student = GRUStudentModel(MyDataset.NbFeatures, 20).to(Device)
        
    Criterion = nn.MSELoss()
    Optimizer = torch.optim.Adam(Student.parameters(), lr=LearningRate)

    Model.eval()
    Student.train()
    
    for i in tqdm(range(Epochs)):
        TrainLoss = 0
        for batch, (x,y) in enumerate(MyDataloader):
            x = x.to(Device).float()
            with torch.no_grad():
                TeacherPred = Model(x)
            Studentpred = Student(x)
            Loss = Criterion(Studentpred, TeacherPred)
            TrainLoss += Loss.item()
            Optimizer.zero_grad()
            Loss.backward()
            Optimizer.step()
    
        if Verbose == 1 and (i+1) % 1 == 0:
            print(f'Epoch:{i+1}, Train Loss:{TrainLoss/len(MyDataloader)}')

    return Student

def L2AttackDetection(Model, DetectionModel, X, y, Objective = 0, LearningRate = 0.000018, c = 1, d = 10000, epochs = 1000, Device = "cpu", Verbose = 0):
    Model.train()
    Adv = X.clone().float()
    Adv.requires_grad = True
    Optimizer = optim.Adam([Adv], lr=LearningRate)
    MseLoss = nn.MSELoss()
    
    for i in tqdm(range(epochs)):
        Optimizer.zero_grad()
        Output = Model(Adv)
        OutputDetection = DetectionModel(Adv)
        Target = torch.full_like(y, Objective)
        TargetLoss = MseLoss(Output, Target)
        DetectionModel
        DiffLoss = MseLoss(Adv.float(), X.float())
        DetectionLoss = MseLoss(OutputDetection.float(), torch.tensor(0).to(Device).float())
        Loss = c * TargetLoss +  DiffLoss + d*DetectionLoss
        if Verbose == 1:
            print("Loss", Loss, TargetLoss, DiffLoss, DetectionLoss)
        
        Loss.backward()
        Optimizer.step()
        with torch.no_grad():
            Adv.clamp_(0.0, 1.0)
    return Adv