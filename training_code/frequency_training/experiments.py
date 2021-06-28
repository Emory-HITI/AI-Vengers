import numpy as np
import Constants
    
def combinations(grid):
    keys = list(grid.keys())
    limits = [len(grid[i]) for i in keys]
    all_args = []
    
    index = [0]*len(keys)
    
    while True:
        args = {}
        for c, i in enumerate(index):
            key = keys[c]
            args[key] = grid[key][i]
        all_args.append(args)
        
        # increment index by 1
        carry = False
        index[-1] += 1
        ind = len(index) -1
        while ind >= 0:
            if carry:
                index[ind] += 1
            
            if index[ind] == limits[ind]:
                index[ind] = 0
                carry = True
            else:
                carry = False                 
            ind -= 1
       
        if carry:
            break
        
    return all_args
        
def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].hparams()    


def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].fname
 
    
#### write experiments here    
class IFFT():
    fname = 'train_model.py'
    @staticmethod
    def hparams():
        grid = {
            'domain': ['both', 'MIMIC', 'CXP'],
            'use_pretrained': [True],
            'seed': [0],
            'target': ['race', 'gender', 'Pneumonia'],
            'data_type': ['ifft'],
            'model': ['densenet'],
            'filter_type': ['low', 'high'],
            'filter_thres': [1, 5, 10, 25, 50, 75, 100, 125, 150, 200],
            'add_noise': [False, True],
            'crop_patch_at_end': [False],
            'augment': [False]
        }
        return combinations(grid)

class Normal():
    fname = 'train_model.py'
    @staticmethod
    def hparams():
        grid = {
            'domain': ['both', 'MIMIC', 'CXP'],
            'use_pretrained': [True],
            'seed': [0],
            'target': ['race', 'gender', 'Pneumonia'],
            'data_type': ['normal'],
            'model': ['densenet', 'vision_transformer'],
            'pixel_thres': [1.0, 0.6],
            'augment': [True, False]
        }
        return combinations(grid)

class IFFTPatched():
    fname = 'train_model.py'
    @staticmethod
    def hparams():
        grid = {
            'domain': ['MIMIC'],
            'use_pretrained': [True],
            'seed': [0],
            'target': ['race', 'gender', 'Pneumonia'],
            'data_type': ['ifft', 'normal'],
            'model': ['densenet'],
            'filter_type': ['high'],
            'filter_thres': [100],
            'patch_ind': list(range(9)),
            'add_noise': [False],
            'crop_patch_at_end': [False],
            'augment': [False],
            'patched': ['patch', 'invpatch']
        }
        return combinations(grid)


class IFFTNotchBandpass():
    fname = 'train_model.py'
    @staticmethod
    def hparams():
        grid = {
            'domain': ['MIMIC'],
            'use_pretrained': [True],
            'seed': [0],
            'target': ['race', 'gender', 'Pneumonia'],
            'data_type': ['ifft'],
            'model': ['densenet'],
            'filter_type': ['notch', 'bandpass'],
            'filter_thres': [10, 25, 50, 75, 100, 125, 150],
            'filter_thres2': [10, 25, 50, 75, 100, 125, 150],
            'add_noise': [False],
            'crop_patch_at_end': [False],
            'augment': [False],
        }
        options = combinations(grid)
        
        return [i for i in options if i['filter_thres2'] > i['filter_thres']]
