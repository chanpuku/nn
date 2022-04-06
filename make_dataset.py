import ray
import numpy as np
import os
import time
from vectorise_controller import vectorise_controller




@ray.remote
def make_dataset(input_folder,output_folder,file_names):
    vcon=vectorise_controller()
    er=0
    for file_name in file_names:
        array=np.load(f'{input_folder}/{file_name}')
        for player_id in range(3):
            try:
                data_x,data_y=vcon.transform(array,player_id)
                np.save(f'{output_folder}/x/{file_name}_hidden{player_id}.npy',data_x)
                np.save(f'{output_folder}/y/{file_name}_hidden{player_id}.npy',data_y)
            except:
                er+=1
    return er

def main():
    ray.init()
    input_folder='vector'
    output_folder='dataset'
    file_names=os.listdir(input_folder)
    length=len(file_names)
    M=10**3 #1processが処理するarrayの数
    N=int((length-0.5)//M+1)#processの数
    work_in_progresses = [make_dataset.remote(input_folder,output_folder,file_names[M*i:M*(i+1)]) for i in range(N)]
    start=time.time()
    recode=1
    error=0
    for i in range(N):
        finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
        orf = finished[0]
        er=ray.get(orf)
        error+=er
        if i+1==recode:
            cur_time=time.time()
            sum_time=cur_time-start
            sum_time/=60#m表記
            per_one=sum_time/(M*(i+1))
            finish_time=per_one*length
            print(f'{(M*(i+1))}/{length}, last:{finish_time-sum_time},[{sum_time}<{finish_time}, {1/per_one}it/m],error:{error}')
            if recode<N/4:
                recode*=2
            else:
                recode+=(N//10)
    


if __name__ == "__main__":
    main()