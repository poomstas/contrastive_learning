# %%
import socket
import argparse

import torch
from datetime import datetime

from src.paths import DATA
from src.model import SimCLRPointCloud
from src.util_parse_arguments import parse_arguments, get_dict_from_args

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ShapeNet
from pytorch_metric_learning.losses import NTXentLoss

# %%
class TrainSimCLR(pl.LightningModule):
    def __init__(self,  
                 AUGMENTATIONS                  = T.Compose([T.RandomJitter(0.005), T.RandomFlip(1), T.RandomShear(0.3)]),
                 LR                             = 0.001,
                 BATCH_SIZE                     = 64,
                 NUM_EPOCHS                     = 10000,
                 CATEGORIES                     = ['Table', 'Lamp', 'Guitar', 'Motorbike'],
                 N_DATASET                      = 5000,
                 TEMPERATURE                    = 0.10,
                 STEP_LR_STEP_SIZE              = 20,
                 STEP_LR_GAMMA                  = 0.5,
    ):


        super(TrainSimCLR, self).__init__()
        self.save_hyperparameters()             # Need this later to load_from_checkpoint without providing the hyperparams again

        self.augmentations                      = AUGMENTATIONS
        self.lr                                 = LR
        self.bs                                 = BATCH_SIZE
        self.n_epochs                           = NUM_EPOCHS
        self.categories                         = CATEGORIES
        self.n_dataset                          = N_DATASET

        self.step_lr_step_size                  = STEP_LR_STEP_SIZE
        self.step_lr_gamma                      = STEP_LR_GAMMA

        self.model                              = SimCLRPointCloud(self.augmentations)
        self.loss                               = NTXentLoss(temperature=TEMPERATURE)
        self.loss_cum                           = 0

        print('='*90)
        print('MODEL HYPERPARAMETERS')
        print('='*90)
        print(self.hparams)
        print('='*90)


    def prepare_data(self) -> None:
        self.data = ShapeNet(root=DATA, categories=self.categories).shuffle()[:self.n_dataset]


    def train_dataloader(self):
        return self.dataloader


    def val_dataloader(self): # In this case (unsupervised learning), val_dataloader is the same as train_dataloader
        self.dataloader = DataLoader(dataset        = self.data,
                                     batch_size     = self.bs,
                                     shuffle        = False,
                                     num_workers    = 12,
                                     pin_memory     = True) # pin_memory=True to keep the data in GPU
        return self.dataloader


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size   = self.step_lr_step_size,
                                                    gamma       = self.step_lr_gamma)
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]


    def forward(self, data, train=True):
        if train:
            h1, h2, compact_h1, compact_h2 = self.model(data, train=train)
            return h1, h2, compact_h1, compact_h2
        else:
            max_pool_val = self.model(data, train=train)
            return max_pool_val


    def validation_step(self, batch, batch_idx):
        return


    def training_step(self, batch, batch_idx):
        h1, h2, compact_h1, compact_h2 = self.model(batch, train=True)
        embeddings = torch.cat((compact_h1, compact_h2))
        indices = torch.arange(0, compact_h1.shape[0], device=self.device)
        labels = torch.cat((indices, indices))
        loss = self.loss(embeddings, labels)
        self.loss_cum += loss
        self.log('loss', loss)

        return {'loss': loss}
        
    def on_training_epoch_end(self, outputs):
        self.log('loss_epoch', self.loss_cum)
        self.loss_cum = 0
        return

    def on_fit_end(self):
        return
        

# %%
if __name__=='__main__':
    torch.set_float32_matmul_precision('medium') # medium or high. See: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    hostname            = socket.gethostname()
    print('Hostname: {}'.format(hostname))

    parser = argparse.ArgumentParser(description='Parse hyperparameter arguments from CLI')
    args = parse_arguments(parser) # Reference values like so: args.alpha 
    config = get_dict_from_args(args)

    run_ID = '_'.join([hostname, config['NOTE'], datetime.now().strftime('%Y%m%d_%H%M%S')])

    cb_checkpoint = ModelCheckpoint(dirpath     = './model_checkpoint/{}/'.format(run_ID), 
                                    monitor     = 'val_loss', 
                                    filename    = '{epoch:02d}-{val_loss:.5f}',
                                    save_top_k  = 10)

    trainer = Trainer(
        accelerator                     = 'gpu',
        devices                         = 'auto',
        max_epochs                      = config['NUM_EPOCHS'], 
        log_every_n_steps               = 1,
        fast_dev_run                    = False, # Run a single-batch through train & val and see if the code works
        logger                          = [],
        callbacks                       = [cb_checkpoint])

    model = TrainSimCLR()

    trainer.fit(model)
