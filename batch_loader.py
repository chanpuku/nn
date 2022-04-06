import ray
import numpy as np
import torch.nn.utils.rnn as rnn
import torch


@ray.remote
class batch_loader:
    """
    データ全部はメモリに乗らんから取り出す
    一人を隠す
    入力、出力の補助
    batchの作成
    data_augmentationしたい
    """
    def __init__(self,data_files,config):
        self.step=0
        self.data_files=data_files
        self.batch_size=config.batch_size
        self.data_length=len(data_files)
        self.step_length=self.data_length//self.batch_size
        self.permutation=np.random.permutation(self.data_length)
        
        self.data_x_folder=config.data_x_folder
        self.data_y_folder=config.data_y_folder
       
    

    def make_data(self,i):
        x=np.load(f'{self.data_x_folder}/{self.data_files[i]}')
        y=np.load(f'{self.data_y_folder}/{self.data_files[i]}')

        #x.shape=[l,727]
        #y.shpae=[l,224]
        length=x.shape[0]
        idx=np.random.randint(length)
        x=torch.from_numpy(x[:idx].astype(np.float32))
        y=torch.from_numpy(y[idx].astype(np.float32))
        return x,y
    def make_next_batch(self):
        idxs=self.permutation[self.step*self.batch_size:(self.step+1)*self.batch_size]
        xl=[]
        yl=[]
        for i in range(self.batch_size):
            x,y=self.make_data(i)
            xl.append(x)
            yl.append(y)
        x=rnn.pad_sequence(xl,batch_first=True)
        y=torch.stack(yl)

        self.step+=1
        if self.step==self.step_length:
            self.permutation=np.random.permutation(self.data_length)
            self.step=0
        return x,y
    
    def get_step_length(self):
        return self.step_length
    
   