import torch
import torch.nn as nn

class my_config:
    def __init__(self):
        bce=nn.BCELoss()
        bce_logit=nn.BCEWithLogitsLoss()
        sigmoid=nn.Sigmoid()
        softmax=nn.Softmax(dim=1)
        kl_div=nn.KLDivLoss(reduction="batchmean")

        def my_kl_div(true,pred):
            return kl_div(pred.log(),true)


        def js_div(x,y):
            return (kl_div(x.log(),y)+kl_div(y.log(),x))/2


        #
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f'use device: cuda')
        else:
            self.device = torch.device('cpu')
            print(f'use device: cpu')
        

        #data_set
        self.data_x_folder='dataset/x'
        self.data_y_folder='dataset/y'

        self.num_batch_maker=18



        #train
        self.num_epochs=20
        self.lr=0.0005
        #model

        

        #lstm
        self.batch_size=1024
        self.lstm_input_dim=732
        self.lstm_hidden_dim=1000
        self.lstm_output_dim=223
        #task_split
        self.task_models={}

        #task
        self.yaku_part=['yakuhai_anko','yakuhai_toitsu','tanyao','toitoi','chitoi','pin_hon','pn_chin','sou_hon','sou_chin','kokushi']
        self.task_model_dic={}
        
        #pai
        idx=0
        for i in range(29):
            self.task_model_dic[f'pai{i}']={'output_dim':5,'activation':softmax,'loss':my_kl_div,'y_idxs':list(range(idx+i*5,idx+(i+1)*5))}
        #shuntsu
        idx+=29*5
        self.task_model_dic[f'shuntsu']={'output_dim':14,'activation':None,'loss':bce_logit,'y_idxs':list(range(idx,idx+14))}
        #dora0~12
        idx+=14
        self.task_model_dic[f'dora']={'output_dim':13,'activation':softmax,'loss':my_kl_div,'y_idxs':list(range(idx,idx+13))}
        #yaku_part
        idx+=13
        for i,y in enumerate(self.yaku_part):
            self.task_model_dic[y]={'output_dim':1,'activation':None,'loss':bce_logit,'y_idxs':list(range(idx+i,idx+i+1))}
        #tenpai
        idx+=10
        self.task_model_dic[f'tenpai']={'output_dim':1,'activation':None,'loss':bce_logit,'y_idxs':list(range(idx,idx+1))}
        #han1~13
        idx+=1
        self.task_model_dic[f'han']={'output_dim':13,'activation':softmax,'loss':my_kl_div,'y_idxs':list(range(idx,idx+13))}
        #machi
        idx+=13
        self.task_model_dic[f'machi']={'output_dim':27,'activation':None,'loss':bce_logit,'y_idxs':list(range(idx,idx+27))}
        




        #loss

        self.losses_rog_dic={'pai':[f'pai{i}' for i in range(29)],'shuntsu':['shuntsu'],'dora':['dora'],'tenpai':['tenpai'],'han':['han'],'machi':['machi']}
        for y in self.yaku_part:
            self.losses_rog_dic[y]=[y]
                           
