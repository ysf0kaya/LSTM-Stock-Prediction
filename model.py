import torch
import torch.nn as nn

class LSTMModelDemo(nn.Module):
    def __init__(self):
        super(LSTMModelDemo, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)        
        out = out[:, -1, :]         
        out = self.fc(out)           
        return out
