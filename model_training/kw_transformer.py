import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from kw_transformer_layers import PositionalEncoding
import torch
from torch import nn
from kw_TransformerEncoderLayer import TransformerEncoderLayer
from dataloader import create_dataloader, get_spike_threshold

class TransAm(pl.LightningModule):
    def __init__(self,loss_fn,batch_size=32,feature_size=1,num_layers=1,dropout=0.1,nhead=2,
                 attn_type=None,learning_rate=1e-5,weight_decay=1e-6):
        super(TransAm, self).__init__()
       
        self.model_type = 'Transformer'
        self.attn_type=attn_type
        self.batch_size=batch_size
        self.nhead=nhead
        self.feature_size=feature_size
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.loss_fn = loss_fn
        print(f'[batch_size x feature_size] {batch_size} x {feature_size}\n')   

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        #self.transformer_encoder = Encoder( input_size=50,heads=2, embedding_dim=feature_size, dropout_rate=dropout, N=num_layers)
        self.decoder = nn.Linear(feature_size,1)
        #self.save_hyperparameters("feature_size","batch_size", "learning_rate","weight_decay")   
        self.init_weights()
        self.save_hyperparameters()

        
        #self.threshold = get_spike_threshold()
        #print(self.threshold)
        #self.spike_classification = {}
        #self.spike_classification['TP'] = 0
        #self.spike_classification['FP'] = 0
        #self.spike_classification['FN'] = 0
        #self.spike_classification['TN'] = 0
        

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
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
    
    #def RMSE_loss(self, logits, labels):
    #    return self.loss_fn(logits, labels)
    
    #def on_train_start(self):
    #    self.logger.log_hyperparams({"hp/learning_rate": self.learning_rate, 
    #                                           "hp/batch_size": self.batch_size})
    #    kw_dict=dict()
        
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view([self.batch_size, -1, self.feature_size]) 
        pred = self(x)
        # The actual forward pass is made on the 
        #input to get the outcome pred from the model
        pred = pred.view(-1,1)
        loss = self.loss_fn(pred, y)
        #print(f'[TRAIN] {pred}, {y}')


        self.log('training_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view([self.batch_size, -1, self.feature_size])
        pred = self(x)
        pred = pred.view(-1,1)
        loss = self.loss_fn(pred, y)
        self.log('val_loss', loss, prog_bar=True)
        #print('val_loss:',loss)
        return loss
        
    def test_step(self, test_batch, batch_idx,batch_size=1):
        x, y = test_batch
        
        x = x.view([batch_size, -1, self.feature_size])
        pred = self(x)
        pred = pred.view(-1,1)

        pred = pred ** 4
        y = y ** 4

        loss = self.loss_fn(pred, y)
        self.log('Test loss', loss, prog_bar=True)

        #print(self.threshold, pred, y)
        #if self.threshold < pred and self.threshold < y :
        #    self.spike_classification['TP'] += 1
        #elif self.threshold < pred and self.threshold >= y :
        #    self.spike_classification['FP'] += 1
        #elif self.threshold >= pred and self.threshold < y :
        #    self.spike_classification['FN'] += 1
        #elif self.threshold >= pred and self.threshold >= y :
        #    self.spike_classification['TN'] += 1


        return loss

        
    #    print(len(losses)) ## This will be same as number of validation batches
    def predict_step(self,batch,batch_idx):
        X_batch, Y_batch = batch
        preds = self(X_batch.float())

        return preds ** 4
    
    