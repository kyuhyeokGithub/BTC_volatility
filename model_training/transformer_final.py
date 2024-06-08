import math
import pandas as pd
import numpy as np
import time
from datetime import datetime
import copy
import os
import random
from typing import Optional, Any, Union, Callable, Tuple
import mlflow

import torch
from torch import nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F


import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar

from kw_transformer import TransAm

from kw_multi_head_attention_forward import multi_head_attention_forward
from kw_transformer_layers import PositionalEncoding
from kw_transformer_multihead_attention import MultiheadAttention
from kw_TransformerEncoderLayer import TransformerEncoderLayer
from kw_transformer_functions import calculate_metrics, RMSELoss, RMSPELoss, plot_dataset, inverse_transform, format_predictions, plot_predictions,final_split,final_dataload


@hydra.main(version_base='1.2',config_path="configs", config_name="train.yaml")
def main(cfg):  


    # X_train, X_test, y_train, y_test = final_split(df, 'value', 0.1)

    # train_loader, test_loader, test_loader_one, scaler = final_dataload(cfg.params.batch_size,X_train,X_test, y_train, y_test)

    # feature_size = len(X_train.columns) #input_dim 

    loss_fn = RMSELoss

    # train
    model = TransAm(loss_fn,cfg.params.batch_size, cfg.params.feature_size,cfg.params.num_layers,
                    cfg.params.dropout,cfg.params.nhead,cfg.params.attn_type,
                    cfg.params.lr,cfg.params.weight_decay)


    early_stop_callback = EarlyStopping(monitor="val_loss", patience=cfg.params.patience, verbose=0, 
                                        mode="min")

    
    checkpoint_callback = ModelCheckpoint(dirpath="modelcheckpoint/"+cfg.model_checkpoint.outputdir,filename='{epoch}-{val_loss:.3f}'
                                      ,save_top_k=1, monitor="val_loss")
  
 
    
    #lr_monitor = LearningRateMonitor(logging_interval='step')


    # hyperparameters = dict(num_layers=cfg.params.num_layers,feature_size=feature_size,
    #                       batch_size=cfg.params.batch_size,dropout=cfg.params.dropout,
    #                        nhead=cfg.params.nhead,attn_type=cfg.params.attn_type,learning_rate=cfg.params.lr,
    #                        weight_decay=cfg.params.weight_decay,n_epochs=cfg.params.n_epochs,loss_fn=loss_fn.__name__)

    # mlflow.pytorch.autolog()

    logger = MLFlowLogger(experiment_name='kh_experiment' ,save_dir="./loggercheckpoint",run_name="transformer_normal_2")
    #logger = TensorBoardLogger(save_dir="./loggercheckpoint", version=1, name='kw_tensorboardlogger')
    #logger = TensorBoardLogger(save_dir=".",version=3, name='loggercheckpoint')


    trainer = pl.Trainer(callbacks=[TQDMProgressBar(refresh_rate=10),early_stop_callback,checkpoint_callback], 
                         max_epochs=cfg.params.n_epochs,
                         log_every_n_steps=10,
                         logger=logger,
                         accelerator='cpu', 
                        #  devices=-1
                         )

    # with mlflow.start_run(experiment_id=cfg.mlflow.experiment_id,run_name = cfg.mlflow.run_name) as run:
        # mlflow.log_params(hyperparameters)
    trainer.fit(model)
    
    trainer.test()
    #print(model.spike_classification)
    #TP = model.spike_classification['TP']
    #FP = model.spike_classification['FP']
    #FN = model.spike_classification['FN']
    #TN = model.spike_classification['TN']
    #print(f'------------- SPIKE CLASSIFICATION ---------------')
    #print(f'[Precision] {TP/(TP+FP):.4f}')
    #print(f'[  Recall ] {TP/(TP+FN):.4f}')
    #print(f'[F1-score ] {2*TP/(2*TP+FN+FP):.4f}') 

if __name__ == "__main__":
    main()    