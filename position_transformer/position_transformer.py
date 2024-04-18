from models.transformer import Transformer
import torch
import torch.nn as nn

def train_loop(model, opt, loss_fn, dataloader):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        X, y = batch[:, 0], batch[:, 1]
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:,:-1]
        y_expected = y[:,1:]
        
        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)      
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)

def validation_loop(model, loss_fn, dataloader):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:,:-1]
            y_expected = y[:,1:]
            
            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and src_mask
            pred = model(X, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)      
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)

def fit(model, opt, loss_fn, train_dataloader, epochs):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    # Used for plotting later on
    train_loss_list = []
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print()
        
    return train_loss_list

def validate(model, loss_fn, val_dataloader):
    validation_lost_list = []

    validation_loss = validation_loop(model, loss_fn, val_dataloader)
    validation_loss_list += [validation_loss]

    print(f"Validation loss: {validation_loss:.4f}")

    return validation_loss_list
    
def predict(model, input_sequence, max_length=15, SOS_token=-1, EOS_token=-2):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    model.eval()
    
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, y_input, tgt_mask)
        
        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()

def create_dataloader(data, batch_size=16, path='.'):
    batches = []

    for idx in range(0, len(data.index)):
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size < len(data):
            dp = {'X':data.loc[idx, 'pov_vec'], 'y':np.array([data.loc[i, 'dpos'][0], data.loc[i, 'dpos'][1], data.loc[i, 'dpos'][2], data.loc[i, 'dyaw'], data.loc[i, 'dpitch']], dtype=np.float32)}
            batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))
        else:
            break

    return batches

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(
        num_tokens=4, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    counter = 0

    for file in os.listdir('data'):
        counter +=1
        with tarfile.open(os.path.join('data/train', file), 'r') as tar:
            tar = tarfile.open(os.path.join('data/train', f'{file}.tar.gz'))
            tar.extractall()
            tar.close()
            train_data = pd.read_pickle(os.path.join('data/train', f'{file.split('.')[0]}.pkl'))
            os.remove(os.path.join('data/train', f'{file.split('.')[0]}.pkl'))

        train_dataloader = create_dataloader(train_data)

        train_loss_list = fit(model, opt, loss_fn, train_dataloader, 10)
    
        with open('train_loss_list', 'ab') as fp:
            pickle.dump(train_loss_list, fp)

        if counter % 9 == 0:
            with tarfile.open(os.path.join('data/val', file), 'r') as tar:
                tar = tarfile.open(os.path.join('data/val', f'{file}.tar.gz'))
                tar.extractall()
                tar.close()
                val_data = pd.read_pickle(os.path.join('data/val', f'{file.split'.')[0]}.pkl')
                os.remove(os.path.join('data/val', f'{file.split('.')[0]}.pkl'))
            val_dataloader = create_dataloader(val_data)
            validation_loss_list = validate(model, loss_fn, val_dataloader)
            
            with open('validation_loss_list', 'ab') as fp:
                pickle.dump(validation_loss_list, fp)


if __name__ == "__main__":
    main()