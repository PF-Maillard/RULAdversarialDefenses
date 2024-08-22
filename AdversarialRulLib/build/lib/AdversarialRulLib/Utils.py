from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

def InitModel(model, device):
    model = model.to(device)
    ks = [key for key in model.state_dict().keys() if 'linear' in key and '.weight' in key]
    for k in ks:
        nn.init.kaiming_uniform_(model.state_dict()[k])
    bs = [key for key in model.state_dict().keys() if 'linear' in key and '.bias' in key]
    for b in bs:
        nn.init.constant_(model.state_dict()[b], 0)
        
def PredRmse(tensor1, tensor2):
    mse = torch.mean((tensor1 - tensor2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse

def validation(model, valloader, device):
    model.eval()
    Loss_MSE = nn.MSELoss()
    
    X, y = next(iter(valloader))
    X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
    
    with torch.no_grad():
        y_pred = model(X)
        val_loss = Loss_MSE(y_pred, y).item()
        
    return val_loss
    
def test(MyModel, testloader, device):
    MyModel.eval()
    
    X, y = next(iter(testloader))
    X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
    
    loss_L1 = nn.L1Loss()
    Loss_MSE = nn.MSELoss()
    with torch.no_grad():
        y_pred = MyModel(X)
        test_loss_RMSE = PredRmse(y_pred, y).item()
        test_loss_MSE = Loss_MSE(y_pred, y).item()
        test_loss_L1 = loss_L1(y_pred, y).item()
        
    return test_loss_MSE, test_loss_L1, test_loss_RMSE, y_pred, y

def TrainModel(model, trainloader, valloader,epochs, learning_rate, device):
    
    Criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for i in tqdm(range(epochs)):
        
        L = 0
        model.train()
        for batch, (X,y) in enumerate(trainloader):
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
            
            y_pred = model(X)
            loss = Criterion(y_pred, y)
            L += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (i+1) % 1 == 0:
            val_loss = validation(model, valloader, device)
            model.train()
            print(f'epoch:{i+1}, avg_train_loss:{L/len(trainloader)}, val_loss:{val_loss}')
            
    
def GetInfos(X, y, Objective, AX, model):
    X, AX, y, = X.to(torch.float32), AX.to(torch.float32), y.to(torch.float32)
    rmse_adversarials = PredRmse(X, AX).item()
    output = model(AX)
    average = torch.mean(output.float()).item()
    averagey = torch.mean(y.float()).item()
    rmse_pred = PredRmse(y.float(), output.float()).item()
    extra_info = {
        'RealRUL': averagey,
        'Objective': Objective,
        'PredRUL': average,
        'RMSE_adversarial': rmse_adversarials,
        'RMSE_pred': rmse_pred
    }
    return extra_info
    
def DataLoaderToNumpy(DataLoader):
    Inputs = []
    Labels = []
    for inputs, labels in DataLoader:
        Inputs.append(inputs.numpy())
        Labels.append(labels.numpy())
    return np.concatenate(Inputs), np.concatenate(Labels)

def flatten_sequences(X):
    return X.reshape(X.shape[0], -1)

def DisplayRMSE(Model, X, y, Index):
    pred = Model(X)
    mse = torch.mean((pred - y) ** 2)
    rmse = torch.sqrt(mse)
    print(Index, "RMSE ", rmse.item())
    
def DisplayEnsembleRMSE(Model, X, y, Index):
    XTest = flatten_sequences(X.detach().cpu().numpy())
    pred = Model.predict(XTest)
    mse = np.mean((pred - y.cpu().numpy()) ** 2)
    rmse = np.sqrt(mse)
    print(Index, "RMSE ",  rmse)

def DisplayRMSEAll(Models, ModelNames, LX, y):
    for i in range(len(Models)):
        print(ModelNames[i])
        for j in range(len(LX)):
            DisplayRMSE(Models[i], LX[j].float(), y, j)
        print()
            
def DisplayDetection(Model, LX):
    for i in range(len(LX)):
        DetectionRate = torch.round(Model(LX[i].float())).sum()/len(LX[i].float())
        print(i, "Detection rate", DetectionRate.item())
    print()
    
def DisplayEnsembleModels(Models, ModelNames, LX, y):
    for i in range(len(Models)):
        print(ModelNames[i])
        for j in range(len(LX)):
            DisplayEnsembleRMSE(Models[i], LX[j].float(), y, j)
        print()