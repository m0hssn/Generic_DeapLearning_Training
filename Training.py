import numpy as np
import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


class Trainer(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None
        self.writer = None

        self.losses = []
        self.val_losses = []

        self.total_epochs = 0

        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send model to {device}, \\ instead sending to {self.device}.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_tensorboard(self, name, folder='runs'):
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step_fn(self):
        def preform_training_step(x, y):
            self.model.train()

            yhat = self.model(x)

            loss = self.loss_fn(yhat, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        return preform_training_step

    def _make_val_step_fn(self):
        def preform_val_step_fn(x, y):
            self.model.eval()

            yhat = self.model(x)

            loss = self.loss_fn(yhat, y)

            return loss.item()

        return preform_val_step_fn

    def _mini_batch(self, validation=False):

        if validation:
            dataloader = self.val_loader
            step_fn = self.val_step_fn
        else:
            dataloader = self.train_loader
            step_fn = self.train_step_fn

        if dataloader is None:
            return None

        mini_batch_losses = []

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)

        return loss

    def train(self, n_epochs, seed=43):
        self.set_seed(seed)

        for epoch in range(n_epochs):

            self.total_epochs += 1

            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            with torch.no_grad():

                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                self.writer.add_scalars(main_tag='loss', tag_scalar_dict=scalars, global_step=epoch)

        if self.writer:
            self.writer.flush()

    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses,
            'val_loss': self.val_losses
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):

        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        self.model.train()

    def predict(self, x):
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        self.model.train()

        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        if self.val_loader:
            plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def add_graph(self):
        if self.train_loader and self.writer:
            x_dummy, y_dummy = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_dummy.to(self.device))

    @staticmethod
    def set_seed(seed=42):
        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
