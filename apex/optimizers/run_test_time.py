import torch
import time

def encoder(tensor):
    c = sum([i for i in tensor])
    return c    

device = torch.device('cuda')
total_num = 50 * 1024 * 62
a = torch.rand(total_num).to(device)
start_time = time.time()
temp = [encoder(a[i*62:(i*62 + 62)]) for i in range(50 * 1024)]
end_time = time.time()
print('Overall time is :  ',end_time - start_time)