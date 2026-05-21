from model.definition import StutterAnomalyLSTM
from util.dataset import SpeechDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model():
    model = StutterAnomalyLSTM()

    print('Loading Speech Dataset...')
    dataset = SpeechDataset()
    print('Turning Speech Set into DataLoader...')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print('Prepping training...')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for batch in tqdm(dataloader, total=len(dataloader), desc='Training'):
        x = batch[:, :-1, :] # frames 0-8
        y = batch[:, 1:, :] # frames 1-9

        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), p.MODEL_WEIGHTS_PATH)

if __name__ == '__main__':
    train_model()