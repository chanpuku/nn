from torch.utils.tensorboard import SummaryWriter

class TensorBoadLogger:
    def __init__(self,save_dir):
        self.writer = SummaryWriter(log_dir=save_dir)
        # 指標の履歴
    def record(self,loss_dic,step,folder=''):
        for k,v in loss_dic.items():
            self.writer.add_scalar(f'{k}{folder}',v,step)
    def close(self):
        self.writer.close()

      