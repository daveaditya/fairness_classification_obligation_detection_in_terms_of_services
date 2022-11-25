import torch
import torch.nn as nn
from torch.autograd import Variable
import warnings

warnings.filterwarnings("ignore")


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):

        super(GRUModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # RNN
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim))

        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
