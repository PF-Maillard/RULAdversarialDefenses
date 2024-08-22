from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class data(Dataset):
    def __init__(self, list_indices, df, window):
        self.indices = list_indices
        self.df = df
        self.window = window
        
    def __len__(self):
        
        return len(self.indices)
    
    def __getitem__(self, idx):
        
        ind = self.indices[idx]
        X_ = self.df.iloc[ind : ind + self.window, :].drop(['time','unit','rul'], axis = 1).copy().to_numpy()
        y_ = self.df.iloc[ind + self.window - 1]['rul']
        
        return X_, y_
    
    def __getdf__(self, idx):
        
        ind = self.indices[idx]
        df = self.df.iloc[ind : ind + self.window, :].copy()
        
        return df
    
class test(Dataset):
    
    def __init__(self, df, window):
        
        UnitBySize = df.groupby('unit')['time'].max().tolist()
        Uniques = df['unit'].unique().tolist()
        Units = []
        for i in range(len(UnitBySize)):
            if UnitBySize[i] > window:
                Units.append(Uniques[i])
        
        self.units = Units
        self.df = df
        self.window = window
        
    def __len__(self):
        return len(self.units)
    
    def __getitem__(self, idx):
       
        n = self.units[idx]
        U = self.df[self.df['unit'] == n].copy()
        X_ = U.reset_index().iloc[-self.window:,:].drop(['index','unit','time','rul'], axis = 1).copy().to_numpy()
        y_ = U['rul'].min()
        
        return X_, y_
    
    def __getdf__(self, idx):
        
        n = self.units[idx]
        U = self.df[self.df['unit'] == n].copy()
        df = U.reset_index().iloc[-self.window:,:].copy()
        
        return df

def SetDatasets(df_train, df_test, rul_test):
    col_names = []
    rul_list = []
    
    col_names.append('unit')
    col_names.append('time')

    for i in range(1,4):
        col_names.append('os'+str(i))
    for i in range(1,22):
        col_names.append('s'+str(i))

    df_train = df_train.iloc[:,:-2].copy()
    df_train.columns = col_names

    df_test = df_test.iloc[:,:-2].copy()
    df_test.columns = col_names
    
    rul_list = []
    for n in np.arange(1,101):
        
        time_list = np.array(df_train[df_train['unit'] == n]['time'])
        length = len(time_list)
        rul = list(length - time_list)
        rul_list += rul
    df_train['rul'] = rul_list

    rul_list = []
    for n in np.arange(1,101):
        time_list = np.array(df_test[df_test['unit'] == n]['time'])
        length = len(time_list)
        rul_val = rul_test.iloc[n-1].item()
        rul = list(length - time_list + rul_val)
        rul_list += rul
    df_test['rul'] = rul_list
    
    return df_train, df_test, rul_test

def find_smallest_unit(column):
    decimal_places = column.apply(lambda x: len(str(abs(x)).split('.')[1]) if '.' in str(abs(x)) else 0)
    #print(decimal_places)
    non_integer_decimal_places = decimal_places[decimal_places > 0]
    Max = non_integer_decimal_places.max()
    if not Max:
        return 1  
    return Max

def GetMinMaxDictionnary(df):
    minmax_dict = {}
    for c in df.columns:
        if 's' in c:
            smallest_unit = find_smallest_unit(df[c])
            minmax_dict[c+'unit'] = smallest_unit
            minmax_dict[c+'min'] = df[c].min()
            minmax_dict[c+'max']=  df[c].max()
    return minmax_dict

def NormalizeDataset(df, minmax_dict):  
    df_copy = df.copy()     
    for c in df_copy.columns:
        if 's' in c:
            if (minmax_dict[c+'max'] != minmax_dict[c+'min']):
                df_copy[c] = (df_copy[c] - minmax_dict[c+'min']) / (minmax_dict[c+'max'] - minmax_dict[c+'min'])
            else:
                df_copy[c] = 0
    return df_copy

def UnnormalizeDataset(df, minmax_dict):   
    df_copy = df.copy()     
    for c in df_copy.columns:
        if 's' in c:
            df_copy[c] = df_copy[c] * (minmax_dict[c+'max'] - minmax_dict[c+'min']) + minmax_dict[c+'min']
    return df_copy
   
def UnormalizedLoader(Loader, minmax_dict):
    ListDF = []
    for i in range(len(Loader.dataset)):
        CurrentDF = Loader.dataset.__getdf__(0)
        CurrentDF = UnnormalizeDataset(CurrentDF, minmax_dict)
        ListDF.append(CurrentDF) 
    return ListDF