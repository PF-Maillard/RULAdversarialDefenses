import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import copy

from . import Utils as UtilsTool
from . import Attacks as AttacksTool
from .DefenseModels import EnsembleRoundDefenseModel, RoundDefenseModel, RandomizeDefenseModel, AutoEncoderDefenseModel, Autoencoder, GRUDetectionModel, GRUStudentModel

def AdversarialTraining(Model, TrainLoader, ValidationLoader, TestLoader, Objective = 0, AdversarialMethod = "Fgsm", AttackParameters = None, Epochs = 30, LearningRate = 0.001, Device = "cpu", Verbose = 0):
    NewModel = copy.deepcopy(Model)
    
    if Verbose == 1:
        print("Defense adversarial training")
    
    UtilsTool.InitModel(Model, Device)
    NewModel =  TrainAdversarialModel(NewModel, TrainLoader, ValidationLoader, Objective = Objective, AdversarialMethod = AdversarialMethod, AttackParameters = AttackParameters, Epochs = Epochs, LearningRate = LearningRate, Device = Device, Verbose = Verbose)  
    
    if Verbose == 1:
        Mse, L1, Rrmse, _, _ = UtilsTool.test(NewModel, TestLoader, Device)
        print(f'MSE:{round(Mse,2)}, L1:{round(L1,2)}, RMSE:{round(Rrmse,2)}')
    
    return NewModel

def EnsembleInputSqueezing(Model, TrainLoader, ValidationLoader, TestLoader, MaxCoefficient = 3, Epochs = 30, LearningRate = 0.001, Device = "cpu", Verbose = 0):
    NewModel = copy.deepcopy(Model)
    Model.eval()
    
    if Verbose == 1:
        print("Defense ensemble squeezing")
    
    NewModel = EnsembleRoundDefenseModel(NewModel, MaxCoefficient, Device).to(Device)

    UtilsTool.InitModel(NewModel, Device)
    ClassicalTraining(NewModel, TrainLoader, ValidationLoader, Epochs = Epochs, LearningRate = LearningRate, Device = Device, Verbose = Verbose)  
    
    if Verbose == 1:
        Mse, L1, Rrmse, _, _ = UtilsTool.test(NewModel, TestLoader, Device)
        print(f'MSE:{round(Mse,2)}, L1:{round(L1,2)}, RMSE:{round(Rrmse,2)}')
    
    return NewModel

def InputSqueezing(Model, TrainLoader, ValidationLoader, TestLoader, Coefficient = 0.1, Epochs = 30, LearningRate = 0.001, Device = "cpu", Verbose = 0):
    NewModel = copy.deepcopy(Model)
    Model.eval()
    
    if Verbose == 1:
        print("Defense Squeezing")
    
    NewModel = RoundDefenseModel(NewModel, Coefficient, Device).to(Device)
    
    UtilsTool.InitModel(NewModel, Device)
    ClassicalTraining(NewModel, TrainLoader, ValidationLoader, Epochs = Epochs, LearningRate = LearningRate, Device = Device, Verbose = Verbose)  
    
    if Verbose == 1:
        Mse, L1, Rrmse, _, _ = UtilsTool.test(NewModel, TestLoader, Device)
        print(f'MSE:{round(Mse,2)}, L1:{round(L1,2)}, RMSE:{round(Rrmse,2)}')
    
    return NewModel

def InputRandomization(Model, TrainLoader, ValidationLoader, TestLoader, Coefficient = 0.1, Epochs = 30, LearningRate = 0.001, Device = "cpu", Verbose = 0):
    NewModel = copy.deepcopy(Model)
    Model.eval()
    
    if Verbose == 1:
        print("Defense Input randomization")
    
    NewModel = RandomizeDefenseModel(NewModel, Coefficient, Device).to(Device)

    UtilsTool.InitModel(NewModel, Device)
    ClassicalTraining(NewModel, TrainLoader, ValidationLoader, Epochs = Epochs, LearningRate = LearningRate, Device = Device, Verbose = Verbose)  
    
    if Verbose == 1:
        Mse, L1, Rrmse, _, _ = UtilsTool.test(NewModel, TestLoader, Device)
        print(f'MSE:{round(Mse,2)}, L1:{round(L1,2)}, RMSE:{round(Rrmse,2)}')
    
    return NewModel

def EnsembleMethods(TrainLoader, ValidationLoader, TestLoader, RFEstimators = 100, XgbEstimators = 100, Verbose = 0):
    X_train, y_train = UtilsTool.DataLoaderToNumpy(TrainLoader)
    X_val, y_val = UtilsTool.DataLoaderToNumpy(ValidationLoader)
    X_test, y_test = UtilsTool.DataLoaderToNumpy(TestLoader)
 
    if Verbose == 1:
        print("Defense Ensemble models")
 
    X_train = np.concatenate([X_train, X_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)
   
    XTrain = UtilsTool.flatten_sequences(X_train)
    XTest = UtilsTool.flatten_sequences(X_test)
    
    if Verbose == 1:
        print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
        print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')    

    RFModel = RandomForestRegressor(n_estimators=RFEstimators, random_state=42)
    RFModel.fit(XTrain, y_train)
    
    if Verbose == 1:
        Pred = RFModel.predict(XTest)
        MyRMSE = np.sqrt(mean_squared_error(y_test, Pred))
        print(f'Random Forest test RMSE: {MyRMSE:.2f}')
    
    XGBModel = XGBRegressor(n_estimators=XgbEstimators, random_state=42)
    XGBModel.fit(XTrain, y_train)
    
    if Verbose == 1:
        Pred = XGBModel.predict(XTest)
        MyRMSE = np.sqrt(mean_squared_error(y_test, Pred))
        print(f'XGB test RMSE: {MyRMSE:.2f}')
    
    return RFModel, XGBModel


def TrainAdversarialValidation(Model, ValidationLoader, Objective = 0, AdversarialMethod = "Fgsm", AttackParameters = None, Device = "cpu"):
    Model.eval()
    Criterion = torch.nn.MSELoss()
    ValidationLoss = 0
    for batch, (X,y) in enumerate(ValidationLoader):
        X, y = X.float().to(Device), y.float().to(Device)

        if AdversarialMethod == "Fgsm" and AttackParameters != None:
            Adv = AttacksTool.Fgsm(Model, 300 - Objective, X, y, AttackParameters["Epsilon"], Device)
        elif AdversarialMethod == "Bim" and AttackParameters != None:
            Adv = AttacksTool.Bim(Model, 300 - Objective, X, y, AttackParameters["Epsilon"], AttackParameters["Iterations"], Device)
        elif AdversarialMethod == "CW" and AttackParameters != None:
            Adv = AttacksTool.CW(Model, Objective, X, y, AttackParameters["LearningRate"], AttackParameters["c"], AttackParameters["Iterations"], Device)
        else:
            Adv = X

        Pred = Model(Adv)
        Loss = Criterion(Pred, y)
        ValidationLoss+= Loss.item()
    Model.train()
    return ValidationLoss/len(ValidationLoader)

def TrainAdversarialModel(Model, TrainLoader, ValidationLoader, AdversarialMethod = "Fgsm", Objective = 0, AttackParameters = None, Epochs = 30, LearningRate = 0.001, Device = "cpu", Verbose = 0):
    Criterion = nn.MSELoss()
    Optimizer = torch.optim.Adam(Model.parameters(), lr=LearningRate)
    
    if Verbose == 1:
        print("TRAINING: Model")
    
    if AttackParameters == None:
        print("ERROR: Miss attack parameters")
    
    Model.train()
    L = 0
    for i in tqdm(range(Epochs)):
        TrainLoss = 0
        Model.train()
        for batch, (X,y) in enumerate(TrainLoader):
            
            X, y = X.to(Device).to(torch.float32), y.to(Device).to(torch.float32)
            if AdversarialMethod == "Fgsm" and AttackParameters != None:
                Adv = AttacksTool.Fgsm(Model, 300 - Objective, X, y, AttackParameters["Epsilon"], Device)
            elif AdversarialMethod == "Bim" and AttackParameters != None:
                Adv = AttacksTool.Bim(Model, 300 - Objective, X, y, AttackParameters["Epsilon"], AttackParameters["Iterations"], Device)
            elif AdversarialMethod == "CW" and AttackParameters != None:
                Adv = AttacksTool.CW(Model, Objective, X, y, AttackParameters["LearningRate"], AttackParameters["c"], AttackParameters["Iterations"], Device)
            else:
                Adv = X
            
            X_combined = torch.cat((X, Adv), dim=0)
            y_combined = torch.cat((y, y), dim=0)
            
            y_pred = Model(X_combined)
            Loss = Criterion(y_pred, y_combined)
            L += Loss.item()
            Optimizer.zero_grad()
            Loss.backward()
            Optimizer.step()
        
        
        if Verbose == 1 and (i+1) % 1 == 0:
            ValLoss = TrainAdversarialValidation(Model, ValidationLoader, Objective = Objective, AdversarialMethod = AdversarialMethod, AttackParameters = AttackParameters, Device = Device)
            print(f'Epoch:{i+1}, Train loss:{TrainLoss/len(TrainLoader)}, Validation loss:{ValLoss}')
            
    return Model
   
def AutoEncoderValidation(Model, ValidationLoader, Device):
    Criterion = torch.nn.MSELoss()
    Model.eval()
    ValidationLoss = 0
    for batch, (X,y) in enumerate(ValidationLoader):
        X = X.float().to(Device)
        Pred = Model(X.float())
        Loss = Criterion(X.float(), Pred.float()) 
        ValidationLoss+= Loss.item()/len(ValidationLoader)
    Model.train()
    return ValidationLoss/len(ValidationLoader)    

def GenerateAutoEncoder(Model, TrainLoader, ValidationLoader, RepresentationDimension = 100, Epochs = 30, LearningRate = 0.001, Device = "cpu", Verbose = 0):
    Model.eval()
    Data = next(iter(TrainLoader))
    
    MyAutoEncoder = Autoencoder(Data[0].shape[2] * Data[0].shape[1], RepresentationDimension, Device).to(Device)
    Criterion = nn.MSELoss()
    Optimizer = optim.Adam(MyAutoEncoder.parameters(), lr=LearningRate)

    if Verbose == 1:
        print("TRAINING: AutoEncoder")
        
    for i in tqdm(range(Epochs)):
        TrainLoss = 0.0
        for X, y in TrainLoader:
            X = X.to(Device)
            y = y.to(Device)
            Optimizer.zero_grad()
            Pred = MyAutoEncoder(X.float())
            Loss = Criterion(X.float(), Pred.float())
            Loss.backward()
            Optimizer.step()
            TrainLoss += Loss.item()
            
        if Verbose == 1 and (i+1) % 1 == 0:
            ValLoss = AutoEncoderValidation(MyAutoEncoder, ValidationLoader, Device)
            print(f'Epoch:{i+1}, Train loss:{TrainLoss/len(TrainLoader)}, Validation loss:{ValLoss}')
    NewModel  = AutoEncoderDefenseModel(MyAutoEncoder, Model)  
    return NewModel

def InputPurificationAutoEncoder(Model, TrainLoader, ValidationLoader, TestLoader, EpochsAutoEncoder = 30 , LearningRateAutoEncoder = 0.001, EpochsModel = 30, LearningRateModel = 0.001, Device = "cpu", Verbose = 0):
    NewModel = copy.deepcopy(Model)
    Model.eval()
    
    if Verbose == 1:
        print("Defense AutoEncoder purification")
    
    NewModel = GenerateAutoEncoder(NewModel, TrainLoader, ValidationLoader, RepresentationDimension = 100, Epochs = EpochsAutoEncoder, LearningRate = LearningRateAutoEncoder, Device = Device, Verbose = Verbose)

    UtilsTool.InitModel(NewModel, Device)
    ClassicalTraining(NewModel, TrainLoader, ValidationLoader, Epochs = EpochsModel, LearningRate = LearningRateModel, Device = Device, Verbose = Verbose)  
    
    if Verbose == 1:
        Mse, L1, Rrmse, _, _ = UtilsTool.test(NewModel, TestLoader, Device)
        print(f'MSE:{round(Mse,2)}, L1:{round(L1,2)}, RMSE:{round(Rrmse,2)}')
    
    return NewModel
    
def DetectionValidation(DetectionModel, Model, ValidationLoader, Objective, Epsilon, Device, AdversarialMethod = "Fgsm", AttackParameters= None):
    Criterion = torch.nn.MSELoss()
    DetectionModel.eval()
    Model.eval()
    ValidationLoss = 0
    for batch, (X,y) in enumerate(ValidationLoader):
        X, y = X.float().to(Device), y.float().to(Device)
        
        if AdversarialMethod == "Fgsm" and AttackParameters != None:
            Adv = AttacksTool.Fgsm(Model, 300 -Objective, X, y, AttackParameters["Epsilon"], Device)
        elif AdversarialMethod == "Bim" and AttackParameters != None:
            Adv = AttacksTool.Bim(Model, 300 -Objective, X, y, AttackParameters["Epsilon"], AttackParameters["Iterations"], Device)
        elif AdversarialMethod == "CW" and AttackParameters != None:
            Adv = AttacksTool.CW(Model, Objective, X, y, AttackParameters["LearningRate"], AttackParameters["c"], AttackParameters["Iterations"], Device)
        else:
            Adv = X
        
        y_X = torch.zeros(X.size(0), dtype=torch.float32)
        y_Adv = torch.ones(Adv.size(0), dtype=torch.float32)
        
        data = torch.cat((X, Adv), dim=0).to(Device)
        labels = torch.cat((y_X, y_Adv), dim=0).to(Device) 
        
        Pred = DetectionModel(data)
        Loss = Criterion(Pred, labels)
        ValidationLoss+= Loss.item()
    DetectionModel.train()
    return ValidationLoss/len(ValidationLoader)   
    
def GenerateDetectionModel(Model, TrainLoader, ValidationLoader, AdversarialMethod = "Fgsm", AttackParameters= None, Objective = 0, Epochs = 30, LearningRate = 0.001, Device = "cpu", Verbose = 0):
    
    Model.eval()
    Data = next(iter(TrainLoader))
    
    input_size = Data[0].shape[2] 
    hidden_size = 128 
    num_layers = 8    

    Detectionmodel = GRUDetectionModel(input_size, hidden_size, num_layers).to(Device)

    Criterion = nn.MSELoss()
    Optimizer = torch.optim.Adam(Detectionmodel.parameters(), lr=LearningRate)

    if Verbose == 1:
        print("TRAINING: Model")

    Detectionmodel.train()
    for i in tqdm(range(Epochs)):
        TrainLoss = 0
        for batch, (X,y) in enumerate(TrainLoader):
            X, y = X.to(Device).to(torch.float32), y.to(Device).to(torch.float32)
            if AdversarialMethod == "Fgsm" and AttackParameters != None:
                Adv = AttacksTool.Fgsm(Model, 300 - Objective, X, y, AttackParameters["Epsilon"], Device)
            elif AdversarialMethod == "Bim" and AttackParameters != None:
                Adv = AttacksTool.Bim(Model, 300 - Objective, X, y, AttackParameters["Epsilon"], AttackParameters["Iterations"], Device)
            elif AdversarialMethod == "CW" and AttackParameters != None:
                Adv = AttacksTool.CW(Model, Objective, X, y, AttackParameters["LearningRate"], AttackParameters["c"], AttackParameters["Iterations"], Device)
            else:
                Adv = X
            
            y_X = torch.zeros(X.size(0), dtype=torch.float32)
            y_Adv = torch.ones(Adv.size(0), dtype=torch.float32)
            
            Data = torch.cat((X, Adv), dim=0)
            Labels = torch.cat((y_X, y_Adv), dim=0)
            
            y_pred = Detectionmodel(Data)
            Loss = Criterion(y_pred.to(Device), Labels.to(Device))
            TrainLoss += Loss.item()
            Optimizer.zero_grad()
            Loss.backward()
            Optimizer.step()
        
        if Verbose == 1 and (i+1) % 1 == 0:
            ValLoss = DetectionValidation(Detectionmodel, Model, ValidationLoader, Objective, Device, AdversarialMethod = AdversarialMethod, AttackParameters = AttackParameters)
            print(f'Epoch:{i+1}, Train loss:{TrainLoss/len(TrainLoader)}, Validation loss:{ValLoss}')
            
    return Detectionmodel

def AdversarialDetection(Model, TrainLoader, ValidationLoader, TestLoader, AdversarialMethod = "Fgsm", AttackParameters= None, Objective = 0, Epochs = 30, LearningRate = 0.001, Device = "cpu", Verbose = 0):
    Model.eval()

    if Verbose == 1:
        print("Defense adversarial Detection")

    DetectionModel = GenerateDetectionModel(Model, TrainLoader, ValidationLoader, Objective = Objective, AdversarialMethod = AdversarialMethod, AttackParameters = AttackParameters, Epochs = Epochs, LearningRate = LearningRate, Device = Device, Verbose = Verbose )
    
    if Verbose == 1:
        Mse, L1, Rrmse, _, _ = UtilsTool.test(DetectionModel, TestLoader, Device)
        print(f'MSE:{round(Mse,2)}, L1:{round(L1,2)}, RMSE:{round(Rrmse,2)}')

    return DetectionModel

def GenerateStudentModel(Model, TrainLoader, ValidationLoader, Epochs =30, LearningRate = 0.001, Device = "cpu", Verbose = 0):
    Data = next(iter(TrainLoader))
    n_features = Data[0].shape[2]
    
    Student = GRUStudentModel(n_features, 20).to(Device)

    Criterion = nn.MSELoss()
    Optimizer = torch.optim.Adam(Student.parameters(), lr=LearningRate)

    if Verbose == 1:
        print("TRAINING: Model")

    Model.eval()
    Student.train()
    for i in tqdm(range(Epochs)):
        TrainLoss = 0
        for _, (x,y) in enumerate(TrainLoader):
            x, y = x.to(Device).to(torch.float32), y.to(Device).to(torch.float32)
            with torch.no_grad():
                TeacherPred = Model(x.float())
            Studentpred = Student(x.float())
            Loss = Criterion(Studentpred, TeacherPred)
            TrainLoss += Loss.item()
            Optimizer.zero_grad()
            Loss.backward()
            Optimizer.step()
        
        if Verbose == 1 and (i+1) % 1 == 0:
            ValLoss = UtilsTool.validation(Student, ValidationLoader, Device)
            Student.train()
            print(f'Epoch:{i+1}, Train loss:{TrainLoss/len(TrainLoader)}, Validation loss:{ValLoss}')

    return Student
    
def DefensiveDistillation(Model, TrainLoader, ValidationLoader, TestLoader, Epochs = 30, LearningRate = 0.001, Device = "cpu", Verbose = 0):        
    NewModel = copy.deepcopy(Model)
    Model.eval()
    
    if Verbose == 1:
        print("Defense Distillation")
        
    NewModel = GenerateStudentModel(NewModel, TrainLoader, ValidationLoader, Epochs = Epochs, LearningRate = LearningRate, Device = Device, Verbose = Verbose)

    if Verbose == 1:
        Mse, L1, Rrmse, _, _ = UtilsTool.test(NewModel, TestLoader, Device)
        print(f'MSE:{round(Mse,2)}, L1:{round(L1,2)}, RMSE:{round(Rrmse,2)}')
        
    return NewModel

def ClassicalTraining(Model, TrainLoader, ValidationLoader, Epochs = 30, LearningRate = 0.001, Device = "cpu", Verbose = 0):
    
    Criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=LearningRate)
    
    for i in tqdm(range(Epochs)):
        
        TrainLoss = 0
        Model.train()
        for batch, (X,y) in enumerate(TrainLoader):
            X, y = X.to(Device).to(torch.float32), y.to(Device).to(torch.float32)
            
            y_pred = Model(X)
            Loss = Criterion(y_pred, y)
            TrainLoss += Loss.item()
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
        
        if Verbose == 1 and (i+1) % 1 == 0:
            ValLoss = ClassicalValidation(Model, ValidationLoader, Device = Device)
            Model.train()
            print(f'Epoch:{i+1}, Train loss:{TrainLoss/len(TrainLoader)}, Validation loss:{ValLoss}')
            
def ClassicalValidation(Model, ValidationLoader, Device = "cpu"):
    Model.eval()
    Loss_MSE = nn.MSELoss()
    
    X, y = next(iter(ValidationLoader))
    X, y = X.to(Device).to(torch.float32), y.to(Device).to(torch.float32)
    
    with torch.no_grad():
        y_pred = Model(X)
        val_loss = Loss_MSE(y_pred, y).item()
        
    return val_loss