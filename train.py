# %%
import socket
import argparse
import wandb

import torch
from torch.utils.data import DataLoader
from datetime import datetime

from src.data import DatasetClass
from src.model import TemporalConvNetSecond
from src.util_parse_arguments import parse_arguments, get_dict_from_args, parse_list_from_wandb

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# %%
class TrainModel(pl.LightningModule):
    def __init__(self,  
                 HYPERPARAM_A       = None,
                 HYPERPARAM_B       = None,
                 HYPERPARAM_C       = None,
                 LR                 = None,
                 NUM_EPOCHS         = 100,
                 BATCH_SIZE         = 32,

                 ONECYCLELR_MAX_LR               = 0.0005,
                 ONECYCLELR_PCT_START            = 0.3,     # The percentage of the cycle spent increasing the learning rate Default: 0.3
                 ONECYCLELR_DIV_FACTOR           = 25,      # Determines the initial learning rate via initial_lr = max_lr/div_factor Default: 25
                 ONECYCLELR_FINAL_DIV_FACTOR     = 0.08,    # Determines the minimum learning rate via min_lr = initial_lr/final_div_factor Default: 1e4
    ):


        super(TrainModel, self).__init__()
        self.save_hyperparameters() # Need this later to load_from_checkpoint without providing the hyperparams again

        self.HYPERPARAM_A = HYPERPARAM_A
        self.HYPERPARAM_B = HYPERPARAM_B
        self.HYPERPARAM_C = HYPERPARAM_C

        self.ONECYCLELR_MAX_LR                  = ONECYCLELR_MAX_LR
        self.ONECYCLELR_PCT_START               = ONECYCLELR_PCT_START
        self.ONECYCLELR_DIV_FACTOR              = ONECYCLELR_DIV_FACTOR
        self.ONECYCLELR_FINAL_DIV_FACTOR        = ONECYCLELR_FINAL_DIV_FACTOR

        self.model                              = TemporalConvNetSecond(model_argument = self.FIRST_TCN_MODEL)
        self.loss                               = LossFunction()
        self.lr                                 = LR

        print('='*90)
        print('MODEL HYPERPARAMETERS')
        print('='*90)
        print(self.hparams)
        print('='*90)


    def prepare_data(self) -> None:
        self.dataset_train = DatasetClass(dataset_argument       = self.SHIFT_JALI_BY,
                                          standardize_JALI_data  = True)

        self.dataset_val   = DatasetClass(dataset_argument       = self.SHIFT_JALI_BY,
                                          standardize_JALI_data  = True)


    def train_dataloader(self):
        self.train_loader = DataLoader(dataset      = self.dataset_train,
                                       batch_size   = self.BATCH_SIZE,

                                       num_workers  = 12,
                                       pin_memory   = True) # pin_memory=True to keep the data in GPU
        return self.train_loader


    def val_dataloader(self):
        self.val_loader = DataLoader(dataset        = self.dataset_val,
                                     batch_size     = self.BATCH_SIZE,
                                     shuffle        = False,
                                     num_workers    = 12,
                                     pin_memory     = True) # pin_memory=True to keep the data in GPU
        return self.val_loader


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        epochs              = self.NUM_EPOCHS,
                                                        steps_per_epoch     = int(round(1.1*len(self.dataset_train)/self.BATCH_SIZE)), # Stretch the schedule extra 10% to prevent bug. See: https://discuss.pytorch.org/t/lr-scheduler-onecyclelr-causing-tried-to-step-57082-times-the-specified-number-of-total-steps-is-57080/90083/3
                                                        max_lr              = self.ONECYCLELR_MAX_LR,
                                                        pct_start           = self.ONECYCLELR_PCT_START,
                                                        div_factor          = self.ONECYCLELR_DIV_FACTOR,
                                                        final_div_factor    = self.ONECYCLELR_FINAL_DIV_FACTOR)
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]


    def forward(self, audio_features):
        jali_xy, rig_params, rig_params_active, _, _ = self.model(audio_features)
        return jali_xy, rig_params, rig_params_active


    def validation_step(self, batch, batch_idx):
        audio_features, jali_xy_target, rig_params_target, rig_params_active_target, _ = batch
        jali_xy_pred, rig_params_pred, rig_params_active_pred = self.forward(audio_features)

        loss_combined = \
            self.loss(jali_xy_target              = jali_xy_target,
                      rig_params_active_pred      = rig_params_active_pred)

        self.log('val_loss',            loss_combined)

        return {'val_loss': loss_combined}


    def validation_epoch_end(self, outputs):
        return

    def training_step(self, batch, batch_idx):
        audio_features, jali_xy_target, rig_params_target, rig_params_active_target, _ = batch
        jali_xy_pred, rig_params_pred, rig_params_active_pred = self.forward(audio_features)

        loss_combined = \
            self.loss(jali_xy_target              = jali_xy_target,
                      rig_params_active_pred      = rig_params_active_pred)

        self.log('loss',                loss_combined)

        return {'loss': loss_combined}
        
    def training_epoch_end(self, outputs):
        return

    def on_fit_end(self):
        # Calculate metrics here and log them on W&B
        return
        

# %%
if __name__=='__main__':
    hostname            = socket.gethostname()
    print('Hostname: {}'.format(hostname))

    parser = argparse.ArgumentParser(description='Parse hyperparameter arguments from CLI')
    args = parse_arguments(parser) # Reference values like so: args.alpha 
    config = get_dict_from_args(args)

    run_ID = hostname if config['NOTE'] == '' else '_'.join([hostname, config['NOTE']])
    run_ID = '_'.join(['SecondTCN', datetime.now().strftime('%Y%m%d_%H%M%S'), run_ID])
    print('='*90)
    print('Run ID: {}'.format(run_ID))
    print('='*90)
    devices = [1] if hostname == 'aorus' else -1

    WANDB_STATE = 'online' # 'online' or 'disabled'
    wandb.init(config=config, project='WandB_Project_Name', entity='poomstas', name=run_ID, mode=WANDB_STATE)
    wandb.config = config # For retrieving hyperparam values from W&B
    logger_wandb = WandbLogger(project='WandB_Project_Name', save_dir='./lightning_logs/', name=run_ID, mode=WANDB_STATE)

    cb_checkpoint = ModelCheckpoint(dirpath     = './model_checkpoint/{}/'.format(run_ID), 
                                    monitor     = 'val_loss', 
                                    filename    = '{epoch:02d}-{val_loss:.5f}',
                                    save_top_k  = 10)

    cb_lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Log code to W&B using Artifacts
    model_code = wandb.Artifact('2nd_tcn_code',         type='training_related_code')
    model_code.add_file('./src/model_second_TCN.py',    name='model_second_TCN.py')
    model_code.add_file('./src/data_second_model.py',   name='data_second_model.py')
    wandb.run.log_artifact(model_code)

    trainer = Trainer(
        accelerator                     = 'gpu',
        devices                         = devices,
        max_epochs                      = config['NUM_EPOCHS'], 
        log_every_n_steps               = 1,
        fast_dev_run                    = False, # Run a single-batch through train & val and see if the code works
        logger                          = [logger_wandb],
        callbacks                       = [cb_checkpoint, cb_lr_monitor],
        auto_lr_find                    = False)

    model = TrainModel(
        ONECYCLELR_MAX_LR               = config['ONECYCLELR_MAX_LR'],
        ONECYCLELR_PCT_START            = config['ONECYCLELR_PCT_START'],
        ONECYCLELR_DIV_FACTOR           = config['ONECYCLELR_DIV_FACTOR'],
        ONECYCLELR_FINAL_DIV_FACTOR     = config['ONECYCLELR_FINAL_DIV_FACTOR'])

    trainer.fit(model)
