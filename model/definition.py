import torch
import torch.nn as nn
import util.paths as p


class StutterAnomalyLSTM(nn.Module):
    def __init__(self, input_size=39, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        prediction = self.fc(lstm_out)
        return prediction

def load_lstm():
    lstm = StutterAnomalyLSTM()
    lstm.load_state_dict(torch.load(p.MODEL_WEIGHTS_PATH))
    return lstm