import torch
from torch.nn.modules import activation
import torchmetrics
import pytorch_lightning as pl

from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=5):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, p, y):
        p_t = y*p + (1-y)*(1-p)
        scale_factor = -(1-p_t)**self.gamma
        losses = scale_factor * torch.log(p_t)
        return losses.mean()


class MeiselClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
    
        self.lstm = nn.LSTM(input_size=6, hidden_size=10, batch_first=True)
        self.hidden = nn.Linear(10, 10) 
        self.activation = nn.ReLU() 
        self.dropout = nn.Dropout(0.7)
        self.output = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

        self.lr = 1e-3
        self.loss = nn.BCELoss()
        self.accuracy = torchmetrics.Accuracy(threshold=0.5)
        self.example_input_array = torch.rand((1, 120, 6))

    def forward(self, x):
        _, (lstm_output, _)  = self.lstm(x)
        hidden = self.hidden(lstm_output.squeeze())
        hidden = self.activation(hidden)
        output = self.output(hidden)
        output = self.sigmoid(output)
        output = output.squeeze()
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        _, (lstm_output, _)  = self.lstm(x)
        hidden = self.hidden(lstm_output.squeeze())
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.output(hidden)
        output = self.sigmoid(output)
        pred = output.squeeze()
        loss = self.loss(pred, y)
        self.log("Training Loss", loss)
        self.log('Training Accuracy', self.accuracy(pred, y.int()))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, (lstm_output, _)  = self.lstm(x)
        hidden = self.hidden(lstm_output.squeeze())
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.output(hidden)
        output = self.sigmoid(output)
        pred = output.squeeze()
        loss = self.loss(pred, y)
        self.log("Validation Loss", loss)
        self.log('Validation Accuracy', self.accuracy(pred, y.int()))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class FocalClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
    
        self.lstm = nn.LSTM(input_size=6, hidden_size=10, batch_first=True)
        self.hidden = nn.Linear(10, 10) 
        self.activation = nn.ReLU() 
        self.dropout = nn.Dropout(0.7)
        self.output = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

        self.lr = 1e-3
        self.loss = FocalLoss()
        self.accuracy = torchmetrics.Accuracy(threshold=0.5)
        self.example_input_array = torch.rand((1, 120, 6))

    def forward(self, x):
        _, (lstm_output, _)  = self.lstm(x)
        hidden = self.hidden(lstm_output.squeeze())
        hidden = self.activation(hidden)
        output = self.output(hidden)
        output = self.sigmoid(output)
        output = output.squeeze()
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        _, (lstm_output, _)  = self.lstm(x)
        hidden = self.hidden(lstm_output.squeeze())
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.output(hidden)
        output = self.sigmoid(output)
        pred = output.squeeze()
        loss = self.loss(pred, y)
        self.log("Training Loss", loss)
        self.log('Training Accuracy', self.accuracy(pred, y.int()))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, (lstm_output, _)  = self.lstm(x)
        hidden = self.hidden(lstm_output.squeeze())
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.output(hidden)
        output = self.sigmoid(output)
        pred = output.squeeze()
        loss = self.loss(pred, y)
        self.log("Validation Loss", loss)
        self.log('Validation Accuracy', self.accuracy(pred, y.int()))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer