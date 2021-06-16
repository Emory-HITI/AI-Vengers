import getpass
import os
import torch
from pathlib import Path
import numpy as np
import math

class EarlyStopping:
    # adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, step, state_dict, path):  # lower loss is better
        score = -val_loss  # higher score is better

        if self.best_score is None:
            self.best_score = score
            self.step = step
            save_model(state_dict, path)
        elif score < self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            save_model(state_dict, path)
            self.best_score = score
            self.step = step
            self.counter = 0
            
def save_model(state_dict, path):
    torch.save(state_dict, path)      

def save_checkpoint(model, optimizer, scheduler, sampler_dict, start_step, es, rng):   
    slurm_job_id = os.environ.get('SLURM_JOB_ID')        
    
    if slurm_job_id is not None and Path('/checkpoint/').exists():        
        torch.save({'model_dict': model.state_dict(),
                    'optimizer_dict': optimizer.state_dict(),
                    'scheduler_dict': scheduler.state_dict(),
                    'sampler_dict': sampler_dict,
                    'start_step': start_step,
                    'es': es,
                    'rng': rng
        } 
                   , 
                   Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt').open('wb')                  
                  )
        
        
def has_checkpoint():
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id is not None and Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt').exists():
        return True
    return False       


def load_checkpoint():   
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id is not None and Path('/checkpoint/').exists():
        return torch.load(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt')
    
def delete_checkpoint():   
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    chkpt_file = Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt')
    if slurm_job_id is not None and chkpt_file.exists():
        return chkpt_file.unlink()   
            
def fft(img):    
    assert(img.ndim == 2)
    img_c2 = np.fft.fft2(img)
    img_c3 = np.fft.fftshift(img_c2)
    spectra = np.log(1+np.abs(img_c3))
    return spectra


def filter_circle(TFcircleIN,fft_img_channel):
    temp = np.zeros(fft_img_channel.shape[:2],dtype=complex)
    temp[TFcircleIN] = fft_img_channel[TFcircleIN]
    return(temp)

def draw_circle(shape,diameter):
    assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diameter **2
    return(TF)

def filter_and_ifft(x, filter):
    return np.real_if_close(np.fft.ifft2(np.fft.ifftshift(filter_circle(filter, x))))

def random_attack(image, epsilon):
    noise = torch.normal(mean=0., std=epsilon, size=image.shape)
    perturbed_image = image + noise
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return noise, perturbed_image

def split_tensor(tensor, tile_size=256, offset=256):
    tiles = []
    h, w = tensor.size(1), tensor.size(2)
    for y in range(int(math.ceil(h/offset))):
         for x in range(int(math.ceil(w/offset))):
             tiles.append(tensor[:, offset*y:min(offset*y+tile_size, h), offset*x:min(offset*x+tile_size, w)])
    if tensor.is_cuda:
         base_tensor = torch.zeros(tensor.size(), device=tensor.get_device())
    else: 
         base_tensor = torch.zeros(tensor.size())
    return tiles, base_tensor

def blacken_tensor(tensor, patch_ind, tile_size=256, offset=256):
    h, w = tensor.size(1), tensor.size(2)
    c = 0
    for y in range(int(math.floor(h/offset))):
        for x in range(int(math.floor(w/offset))):
            if c == patch_ind:
                tensor[:, offset*y:min(offset*y+tile_size, h), offset*x:min(offset*x+tile_size, w)] = 0
            c += 1
    return tensor      