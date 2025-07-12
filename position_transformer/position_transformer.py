from models.transformer import Transformer
import torch
import torch.nn as nn
# import os
# import pandas as pd
from torch.utils.data import DataLoader#, Dataset
from models import PositionDataset
from tqdm import tqdm
import pickle 

BATCH_SIZE = 32

def train_loop(model, device, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0
    
    with tqdm(total=len(dataloader), desc=f"Training... ", position=0) as pbar:
        for batch in dataloader:
            X, y = batch['X'], batch['y']
            X, y = X.clone().detach().to(device), y.clone().detach().to(device)

            y_input = y[:,:-1]
            y_input = torch.cat((torch.zeros((y_input.shape[0], 1, y_input.shape[2]), dtype=torch.float32, device=device), y_input), dim=1)
            y_expected = y[:,1:]

            pred = model(X, y_input)

            loss = loss_fn(pred[:, 1:], y_expected)
            
            pred.to('cpu')
            y_expected.to('cpu')

            opt.zero_grad()
            loss.backward()
            opt.step()
        
            total_loss += loss.detach().item()
            pbar.update(1)
            
    return total_loss / len(dataloader)

def validation_loop(model, device, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    
    with tqdm(total=len(dataloader), desc=f"Validating... ", position=0) as pbar:
        for batch in dataloader:
            X, y = batch['X'], batch['y']
            X, y = X.clone().detach().to(device), y.clone().detach().to(device)

            y_input = y[:,:-1]
            y_input = torch.cat((torch.zeros((y_input.shape[0], 1, y_input.shape[2]), dtype=torch.float32, device=device), y_input), dim=1)
            y_expected = y[:,1:]

            pred = model(X, y_input)

            loss = loss_fn(pred[:, 1:], y_expected)

            pred.to('cpu')
            y_expected.to('cpu')
        
            total_loss += loss.detach().item()
            pbar.update(1)
        
    return total_loss / len(dataloader)

def fit(model, device, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    # Used for plotting later on
    train_loss_list = []
    val_loss_list = []
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, device, opt, loss_fn, train_dataloader)
        train_loss_list.append(train_loss)

        val_loss = validation_loop(model, device, loss_fn, val_dataloader)
        val_loss_list.append(val_loss)
        
        print(f"Training loss: {train_loss}")
        print(f"Validation loss: {val_loss}")
        print()

    torch.save(model.state_dict(), 'model_weights.pt')
        
    return train_loss_list, val_loss_list

def main():
    print('Start model setup')
    print('Starting Transformer model')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Transformer(
        input_dim=2048, feedforward_dim=2048, output_dim=5, num_heads=8, num_layers=3
    ).to(device)
    model.summary(BATCH_SIZE)

    model.load_state_dict(torch.load('model_weights.pt'))

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.HuberLoss()

    train_dataset = PositionDataset('data/processed_data/train')
    val_dataset = PositionDataset('data/processed_data/val')

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_loss_list, val_loss_list = fit(model, device, opt, loss_fn, train_loader, val_loader, 1)

    with open('train_loss_list', 'ab') as fp:
        pickle.dump(train_loss_list, fp)

    with open('val_loss_list', 'ab') as fp:
        pickle.dump(val_loss_list, fp)
    return
        
if __name__ == "__main__":
    main()
    