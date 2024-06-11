import math
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import hydra
import pytorch_lightning as pl

from kw_transformer import TransAm

from kw_transformer_functions import MAELoss, MAPELoss, RMSELoss, RMSPELoss

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
    
    ckpt_path = './modelcheckpoint/workspace/LFD_bitcoin/ckpt/epoch=0-val_loss=0.536.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.freeze()
    
    results = []
    gt = []
    t = model.test_dataloader()
    for j, batch in tqdm(enumerate(t)):
        x, y = batch
        
        x = x.view([-1, cfg.params.feature_size, cfg.params.day_window])
        x = x.transpose(1,2)
        pred = model(x)
        pred = pred.transpose(0,1).squeeze(-1)
        pred = pred[:,0]
        y = y[:,0]
        #print(pred, y)
        pred = pred ** (10/3)
        y = y ** (10/3)
        results.append(pred)
        gt.append(y)
    
    results = torch.stack(results, dim=0)
    gt = torch.stack(gt, dim=0)

    rmse = RMSELoss(results, gt)
    rmspe = RMSPELoss(results, gt)
    mae = MAELoss(results, gt)
    mape = MAPELoss(results, gt)

    print('rmse: ', rmse)
    print('rmspe: ', rmspe)
    print('mae: ', mae)
    print('mape: ', mape)

if __name__ == "__main__":
    main()    