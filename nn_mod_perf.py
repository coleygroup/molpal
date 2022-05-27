import numpy as np  
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as rdmd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def read_data(path, smiles_col, data_cols):
    reader = csv.reader(open(Path(path)))
    data = {}
    for row in reader: 
        try: 
            key = row[smiles_col]
            val = []
            for col in data_cols:
                val.append(float(row[col]))
            data[key]=val
            data[key][0] = -data[key][0] # to make it negative
        except:
            pass
    return data

class fpDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(np.array(X))  # store X as a pytorch Tensor
        self.y = torch.Tensor(np.array(y))  # store y as a pytorch Tensor
        self.len=len(self.X)                # number of samples in the data 

    def __getitem__(self, index):
        return self.X[index], self.y[index] # get the appropriate item

    def __len__(self):
        return self.len

def import_organize_data(train_val_size, col):
    data = read_data('data/selectivity_data.csv', 0, [1,2])
    ys = np.array(list(data.values()))
    smiles = list(data.keys())
    length = 2048
    radius = 2
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fps = [rdmd.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=length, useChirality=True
            ) for mol in mols]
    fps = np.array(fps)
    test_size = 1-(train_val_size/len(smiles))
    X_train_val, X_test, train_val_truth, test_truth = train_test_split(fps,ys,test_size=test_size,random_state=4) 
    X_train, X_val, train_truth, val_truth = train_test_split(X_train_val,train_val_truth,test_size=0.2,random_state=4) 
    train_data = fpDataset(X_train, train_truth[:,col])
    val_data = fpDataset(X_val,val_truth[:,col])
    batch_size = 100; 
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    return test_truth, X_test, train_dataloader, val_dataloader 

class fp_to_score_MLP(torch.nn.Module):
    def __init__(self):
        # You can modify this method to pass hyperparameters above, but this is not necessary
        # since we already have fixed hyperparameters
        super().__init__()
        
        # Implement your code here
        self.model = torch.nn.Sequential(
        torch.nn.Linear(2048,100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1),
        )
        
    def forward(self, x):
        x = self.model(x)
        
        return x

def train(model, dataloader, optimizer, device):

    epoch_loss = []
    model.train() # Set model to training mode 
    
    for batch in dataloader:    
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        
        # train your model on each batch here 
        y_pred = model(X)
        
        loss = torch.nn.functional.mse_loss(y_pred.ravel(),y.ravel())
        epoch_loss.append(loss.item())
        
        # run backpropagation given the loss you defined
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.array(epoch_loss).mean()


def validate(model, dataloader, device):
    
    val_loss = []
    model.eval() # Set model to evaluation mode 
    with torch.no_grad():    
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            
            # validate your model on each batch here 
            y_pred = model(X)

            loss = torch.nn.functional.mse_loss(y_pred.ravel(),y.ravel())
            val_loss.append(loss.item())
            
    return np.array(val_loss).mean()


def run_training(model, train_dataloader, val_dataloader, optimizer):
    val_loss_curve = []
    train_loss_curve = []   
    for epoch in range(25):
        
        # Compute train your model on training data
        epoch_loss = train(model, train_dataloader, optimizer,  device=device)
        
        # Validate your on validation data 
        val_loss = validate(model, val_dataloader, device=device) 
        
        # Record train and loss performance 
        train_loss_curve.append(epoch_loss)
        val_loss_curve.append(val_loss)
        
        # print(epoch, epoch_loss, val_loss)
    return train_loss_curve, val_loss_curve

def model_train(device, train_dataloader, val_dataloader):
    model = fp_to_score_MLP().to(device)
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)
    run_training(model, train_dataloader, val_dataloader, opt)
    return model

def test_performance(model, device, X_test, test_truth, col):
    pred = np.array([model(torch.Tensor(X).to(device)).cpu().detach().numpy() for X in X_test]).squeeze()
    mse = mean_squared_error(test_truth[:,col], pred)
    return mse 

if __name__ == '__main__':
    device = 'cuda:1'
    train_val_sizes = [100, 200, 500, 1000]
    mse = np.zeros((len(train_val_sizes), 2))
    for col in range(2):
        for i in range(len(train_val_sizes)):
            test_truth, X_test, train_dataloader, val_dataloader  = import_organize_data(train_val_sizes[i], col)
            model = model_train(device, train_dataloader, val_dataloader)
            mse[i,col] = test_performance(model, device, X_test, test_truth, col)
    print(mse)