import torch 
import numpy as np

def fgsm_attack(batch, epsilon, data_grads):
    sign_data_grads = data_grads.sign()
    
    perturbed_batch = batch + epsilon*sign_data_grads
    perturbed_batch = torch.clamp(perturbed_batch, 0., 1.)
    
    return perturbed_batch
    
def random_attack(batch, epsilon):
    random_noise = torch.normal(mean = 0., std = 1., size = batch.shape).to(batch.device)
    
    perturbed_batch = batch + epsilon*random_noise
    perturbed_batch = torch.clamp(perturbed_batch, 0., 1.)
    
    return perturbed_batch
