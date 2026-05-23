from model.definition import StutterAnomalyLSTM
from util.dataset import SpeechDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util.paths import MODEL_WEIGHTS_PATH
from tqdm import tqdm


def train_model(epochs=100):
    print('Prepping dataset...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StutterAnomalyLSTM().to(device)

    dataset = SpeechDataset()
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    print('Prepping training...')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print('Entering training loop...')
    for epoch in tqdm(range(epochs), total=epochs, desc='Training'):
        for batch in tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            batch = batch.to(device)
            x = batch[:, :-1, :] # frames 0-8
            y = batch[:, 1:, :] # frames 1-9

            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), str(MODEL_WEIGHTS_PATH).replace('.pt', f'{epoch+1}.pt'))

if __name__ == '__main__':
    train_model()