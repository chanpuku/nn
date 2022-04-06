import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from sklearn.model_selection import train_test_split
import os,datetime
import ray


from config import my_config
import model 
from batch_loader import batch_loader
from pcgrad import PCGrad
from loss_function import loss_controller

from logger import TensorBoadLogger

import time
from tqdm import tqdm
import sys

def main():
    ray.init()
    # GPUを使うために必要
    

    # ------------------ CHANGE THE CONFIGURATION -------------
    data_x=os.listdir('./dataset/x')
    

    date_time=datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    logg_dir=f'result/{date_time}'

    save_model_path='model_{date_time}.pth'
    
    # ---------------------------------------------------------
    
    
    

    config=my_config()
    device=config.device
    my_models=model.my_models(config)
    my_models.to_device(device)
    loss_con=loss_controller(config)

    optimizer = PCGrad(optim.Adam(my_models.parameters(),lr=config.lr)) 

    train_x,test_x=train_test_split(data_x,train_size=0.7)
    

    train_step_length=len(train_x)//config.batch_size
    test_step_length=len(test_x)//config.batch_size
    
    
    
    logger=TensorBoadLogger(logg_dir)

    NUM_EPOCHS = config.num_epochs
    for ep in range(NUM_EPOCHS):
        #train
        files_length=len(train_x)//config.num_batch_maker+1
        batch_loaders=[batch_loader.remote(train_x[i*files_length:(i+1)*files_length],config) for i in range(config.num_batch_maker)]
        batch_return=[batch_loaders[i].make_next_batch.remote() for i in range(config.num_batch_maker)]
        
        
        my_models.train()
        
        
        for i in tqdm(range(train_step_length)):
            j=i%config.num_batch_maker
            bl=batch_loaders[j]
            batch_x,batch_y=ray.get(batch_return[j])
            if i==0:
                print(f'batch_x.shape:{batch_x.shape}')
                print(f'batch_y.shape:{batch_y.shape}')
            optimizer.zero_grad()
            batch_x=batch_x.to(device)
            batch_y=batch_y.to(device)
            pred_y=my_models(batch_x)
            losses=loss_con.losses(pred_y,batch_y)
            optimizer.pc_backward(losses.values())
            optimizer.step()
            

            batch_return[j]=bl.make_next_batch.remote()
            losses_dic=loss_con.rog_transform(losses)
            logger.record(losses_dic,ep*train_step_length+i,folder='/train')
            
            

            
        
        #validation
        files_length=len(test_x)//config.num_batch_maker+1
        batch_loaders=[batch_loader.remote(test_x[i*files_length:(i+1)*files_length],config) for i in range(config.num_batch_maker)]
        batch_return=[batch_loaders[i].make_next_batch.remote() for i in range(config.num_batch_maker)]

        
        my_models.eval()
        

        with torch.no_grad():
            for i in range(test_step_length):
                j=i%config.num_batch_maker
                bl=batch_loaders[j]
                batch_x,batch_y=ray.get(batch_return[j])

                batch_x=batch_x.to(device)
                batch_y=batch_y.to(device)
                pred_y=my_models(batch_x)
                losses=loss_con.losses(pred_y,batch_y)
                

                batch_return[j]=bl.make_next_batch.remote()
                losses_dic=loss_con.rog_transform(losses)
                logger.record(losses_dic,ep*test_step_length+i,folder='/val')

            
    torch.save(my_models.state_dict(), save_model_path)
    logger.close()
            

if __name__ == "__main__":
    main()