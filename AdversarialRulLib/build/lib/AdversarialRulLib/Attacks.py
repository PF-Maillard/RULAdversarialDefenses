import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

def CW(Model, Objective, X, y, LearningRate = 0.01, c = 1, Epochs = 100, Device = "cpu"):
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

def TestAttacks(Model, X, y, AttacksParameters, Device = "cpu"):
    
    X, y = X.to(Device).to(torch.float32), y.to(Device).to(torch.float32)

    AdversarialDataFgsm = Fgsm(Model, AttacksParameters["FGSM"]["Objective"], X, y, Epsilon = AttacksParameters["FGSM"]["Epsilon"], Device = Device)
    AdversarialDataBim = Bim(Model, AttacksParameters["BIM"]["Objective"], X, y, Epsilon = AttacksParameters["BIM"]["Epsilon"], Epochs = AttacksParameters["BIM"]["Iterations"], Device = Device)
    AdversarialDataCW = CW(Model, AttacksParameters["CW"]["Objective"], X, y, LearningRate = AttacksParameters["CW"]["LearningRate"], c = AttacksParameters["CW"]["c"], Epochs = AttacksParameters["CW"]["Iterations"], Device = Device)
    
    Infos = UtilsTool.GetInfos(X, y, 0, AdversarialDataFgsm, Model)
    print("FGSM:", Infos)

    Infos = UtilsTool.GetInfos(X, y, 0, AdversarialDataBim, Model)
    print("BIM:", Infos)

    Infos = UtilsTool.GetInfos(X, y, 0, AdversarialDataCW, Model)
    print("CW:", Infos)
    
    return AdversarialDataFgsm, AdversarialDataBim, AdversarialDataCW

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

def CwAttackDetection(Model, DetectionModel, X, y, Objective = 0, LearningRate = 0.000018, c = 1, d = 10000, epochs = 1000, Device = "cpu", Verbose = 0):
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