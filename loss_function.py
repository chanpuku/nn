import numpy as np
import torch

class loss_controller:
    def __init__(self,config):
        self.config=config
        self.device=config.device
        self.rog_dic=config.losses_rog_dic
        self.task_model_dic=config.task_model_dic

        
    def losses(self,pred_y,true_y):
        losses={}
        for k in pred_y.keys():
            idxs=self.task_model_dic[k]['y_idxs']
            lf=self.task_model_dic[k]['loss']
            losses[k]=lf(true_y[:,idxs],pred_y[k])
        return losses
    def rog_transform(self,losses):
        dic={}
        for k,v in self.rog_dic.items():
            loss_value=0
            for i in v:
                loss_value+=losses[i].item()
            loss_value/=len(v)
            dic[k]=loss_value
        return dic