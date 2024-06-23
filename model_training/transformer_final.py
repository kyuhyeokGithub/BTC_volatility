import math
import pandas as pd
import numpy as np
import mlflow

import torch
from torch import nn
import torch.nn.functional as F


import hydra

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar

from transformer import TransAm

from transformer_functions import  RMSELoss

from dataloader import make_volatility_png

import warnings
warnings.filterwarnings("ignore")

@hydra.main(version_base='1.2',config_path="configs", config_name="train.yaml")
def main(cfg):  

    loss_fn = RMSELoss

    # train
    model = TransAm(loss_fn,cfg.params.batch_size, cfg.params.feature_size,cfg.params.num_layers,
                    cfg.params.dropout,cfg.params.nhead,cfg.params.attn_type,
                    cfg.params.lr,cfg.params.weight_decay, cfg.params.day_window)


    early_stop_callback = EarlyStopping(monitor="val_loss", patience=cfg.params.patience, verbose=0, 
                                        mode="min")

    
    checkpoint_callback = ModelCheckpoint(dirpath="modelcheckpoint/"+cfg.model_checkpoint.outputdir,filename='{epoch}-{val_loss:.3f}'
                                      ,save_top_k=1, monitor="val_loss")
  

    logger = MLFlowLogger(experiment_name='experiment' ,save_dir="./loggercheckpoint",run_name="transformer_normal_2")

    trainer = pl.Trainer(callbacks=[TQDMProgressBar(refresh_rate=10),early_stop_callback,checkpoint_callback], 
                         max_epochs=cfg.params.n_epochs,
                         log_every_n_steps=10,
                         logger=logger,
                         accelerator='cpu', 
                        #  devices=-1
                        )

    make_volatility_png()

    trainer.fit(model)
    
    trainer.test()
    

if __name__ == "__main__":
    main()    