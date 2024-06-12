import pytorch_lightning as pl
from transformer_layers import PositionalEncoding
import torch
from torch import nn
import torch.nn.functional as F
from dataloader import create_dataloader
from transformer_functions import MAELoss, MAPELoss, RMSELoss, RMSPELoss

class TransAm(pl.LightningModule):
    def __init__(self, loss_fn, batch_size=32,feature_size=1,num_layers=1,dropout=0.1,nhead=2,
                 attn_type=None,learning_rate=1e-5,weight_decay=1e-6, day_window=10):
        super(TransAm, self).__init__()
       
        self.model_type = 'Transformer'
        self.attn_type=attn_type
        self.batch_size=batch_size
        self.nhead=nhead
        self.feature_size=feature_size
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.day_window=day_window
        self.loss_fn = loss_fn
        self.loss_fn1 = RMSELoss
        self.loss_fn2 = RMSPELoss
        self.loss_fn3 = MAELoss
        self.loss_fn4 = MAPELoss
        print(f'[batch_size x feature_size] {batch_size} x {feature_size}\n')  

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.fc = nn.Linear(self.day_window, 7)
        self.decoder = nn.Linear(feature_size,1)

        #self.save_hyperparameters("feature_size","batch_size", "learning_rate","weight_decay")   
        self.init_weights()
        self.save_hyperparameters()     

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()    
        self.decoder.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
            self.src_mask = mask
        
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src.transpose(0,1), self.src_mask)#, self.src_mask)
        output = output.transpose(0,2)
        output = self.fc(output)
        output = F.relu(output)
        output = output.transpose(0,2)
        
        output = self.decoder(output)
        #output=F.relu(output)
        #add sigmoid function <- output=sigmoid. force output to be 0-1. and 
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
    def train_dataloader(self):
        # REQUIRED
        # This is an essential function. Needs to be included in the code
               
        return create_dataloader(self.batch_size, 'train')
        
    def val_dataloader(self):
        # OPTIONAL
        #loading validation dataset
        return create_dataloader(self.batch_size, 'valid')

    def test_dataloader(self):
        # OPTIONAL
        # loading test dataset
        return create_dataloader(self.batch_size, 'test')
        
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view([self.batch_size, self.feature_size, -1]) 
        x = x.transpose(1,2)
        pred = self(x)        

        # The actual forward pass is made on the 
        #input to get the outcome pred from the model
        pred = pred.transpose(0,1).squeeze(-1)
        loss = self.loss_fn(pred, y)
        #print(f'[TRAIN] {pred}, {y}')

        self.log('training_loss', loss, prog_bar=True)
        return loss


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view([self.batch_size,self.feature_size, -1])
        x = x.transpose(1,2)        
        pred = self(x)

        pred = pred.transpose(0,1).squeeze(-1)

        loss = self.loss_fn(pred, y)
        self.log('val_loss', loss, prog_bar=True)
        #print('val_loss:',loss)
        return loss
        
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        
        x = x.view([-1, self.feature_size, self.day_window])
        x = x.transpose(1,2)        
        pred = self(x)
        pred = pred.transpose(0,1).squeeze(-1)

        pred = pred ** (10/3)
        y = y ** (10/3)

        mae = self.loss_fn3(pred, y)
        mape = self.loss_fn4(pred, y)

        # self.log('Test loss RMSE', rmse, prog_bar=True)
        # self.log('Test loss RMSPE', rmspe, prog_bar=True)
        self.log('Test loss MAE', mae, prog_bar=True)
        self.log('Test loss MAPE', mape, prog_bar=True)

        return pred, y

        
    #    print(len(losses)) ## This will be same as number of validation batches
    def predict_step(self,batch,batch_idx):
        X_batch, Y_batch = batch
        preds = self(X_batch.float())

        return preds ** (10/3)
    
    