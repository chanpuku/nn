import torch.nn.functional as F
from turtle import forward
import torch.nn as nn

class my_models(nn.Module):
    def __init__(self,config):
        super(my_models, self).__init__()
        self.config=config

        self.output_list=list(config.task_model_dic.keys())
        
        self.lstm=mylstm(config.lstm_input_dim,config.lstm_hidden_dim,config.lstm_output_dim)
        module_dict={}
        for k,v in config.task_model_dic.items():
            module_dict[k]=nn.Linear(config.lstm_output_dim,v['output_dim'])
        self.output_module_dict=nn.ModuleDict(module_dict)
        
    def eval(self):
        self.lstm.eval()
        self.output_module_dict.eval()
    def train(self):
        self.lstm.train()
        self.output_module_dict.train()
    def to_device(self,device):
        self.lstm.to(device)
        self.output_module_dict.to(device)
    def forward(self,input):
        x=self.lstm(input)
        dic={}
        for k in self.output_module_dict.keys():
            y=self.output_module_dict[k](x)
            if not self.config.task_model_dic[k]['activation']==None:
                y=self.config.task_model_dic[k]['activation'](y)
            dic[k]=y
        return dic
    
class mylstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mylstm, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.linear1 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim) #バッチ正規化
        self.bn2 = nn.BatchNorm1d(output_dim) #バッチ正規化
        self.linear2=nn.Linear(output_dim, output_dim)
       

    def forward(self, input):
        #input.size() = (batch_size × len(one_data) × input_dim(=dim(one_data)))
        _, lstm_out = self.lstm(input)
        # lstm_out[0].size() = (1 × batch_size × hidden_dim)
        # (batch_size × tagset_size)にするためにsqueeze()する
        lstm_out=lstm_out[0].squeeze()

        #lstm_out.size() = (batch_size × tagset_size)

        x = self.linear1(lstm_out)
        x= self.bn1(x)
        x=F.relu(x)
        x=self.linear2(x)
        x=self.bn2(x)
        x=F.relu(x)
       

        #このあと
        # linear
        # and
        # activateされる

        return x
