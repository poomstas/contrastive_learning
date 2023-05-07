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

                 ONECYCLELR_MAX_LR              = 0.0005,
                 ONECYCLELR_PCT_START           = 0.3,     # The percentage of the cycle spent increasing the learning rate Default: 0.3
                 ONECYCLELR_DIV_FACTOR          = 25,      # Determines the initial learning rate via initial_lr = max_lr/div_factor Default: 25
                 ONECYCLELR_FINAL_DIV_FACTOR    = 0.08,    # Determines the minimum learning rate via min_lr = initial_lr/final_div_factor Default: 1e4
    ):


        super(TrainSimCLR, self).__init__()
        self.save_hyperparameters()             # Need this later to load_from_checkpoint without providing the hyperparams again

        self.augmentations                      = AUGMENTATIONS
        self.lr                                 = LR
        self.bs                                 = BATCH_SIZE
        self.n_epochs                           = NUM_EPOCHS
        self.categories                         = CATEGORIES
        self.n_dataset                          = N_DATASET

        self.model                              = SimCLRPointCloud(self.augmentations)
        self.loss                               = NTXentLoss(temperature=TEMPERATURE)
        self.loss_cum                           = 0

        self.ONECYCLELR_MAX_LR                  = ONECYCLELR_MAX_LR
        self.ONECYCLELR_PCT_START               = ONECYCLELR_PCT_START
        self.ONECYCLELR_DIV_FACTOR              = ONECYCLELR_DIV_FACTOR
        self.ONECYCLELR_FINAL_DIV_FACTOR        = ONECYCLELR_FINAL_DIV_FACTOR

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
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        epochs              = self.n_epochs,
                                                        steps_per_epoch     = int(round(1.1*len(self.data)/self.bs)), # Stretch the schedule extra 10% to prevent bug. See: https://discuss.pytorch.org/t/lr-scheduler-onecyclelr-causing-tried-to-step-57082-times-the-specified-number-of-total-steps-is-57080/90083/3
                                                        max_lr              = self.ONECYCLELR_MAX_LR,
                                                        pct_start           = self.ONECYCLELR_PCT_START,
                                                        div_factor          = self.ONECYCLELR_DIV_FACTOR,
                                                        final_div_factor    = self.ONECYCLELR_FINAL_DIV_FACTOR)
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
        out = self.model(batch, train=False)
        return out


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

    model = TrainSimCLR(
        ONECYCLELR_MAX_LR               = config['ONECYCLELR_MAX_LR'],
        ONECYCLELR_PCT_START            = config['ONECYCLELR_PCT_START'],
        ONECYCLELR_DIV_FACTOR           = config['ONECYCLELR_DIV_FACTOR'],
        ONECYCLELR_FINAL_DIV_FACTOR     = config['ONECYCLELR_FINAL_DIV_FACTOR'])

    trainer.fit(model)
