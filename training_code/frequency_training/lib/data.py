import torch
import os
import numpy as np
from PIL import Image
import Constants
from lib import cxr_preprocess as preprocess
import pandas as pd
from torchvision import transforms
import pickle
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from lib.utils import fft, filter_and_ifft, draw_circle, random_attack, split_tensor, blacken_tensor
import math


def get_dataset(envs = [], split = None, only_frontal = False, imagenet_norm = True, augment = 0, cache = True, subset_label = None,
               augmented_dfs = None,  output_type = 'normal', ifft_filter = None, add_noise = False,
                    pixel_thres = None, crop_patch_at_end = False, patched = 'none', patch_ind = None):
          
    if augment == 1: # normal image augmentation
        image_transforms = [transforms.RandomHorizontalFlip(), 
                            transforms.RandomRotation(15),     
                            transforms.RandomResizedCrop(size = 224, scale = (0.75, 1.0)),
                        transforms.ToTensor()]        
    elif augment == 0: 
        image_transforms = [transforms.ToTensor()]
    elif augment == -1: # only resize, just return a dataset with PIL images; don't ToTensor()
        image_transforms = []        
   
    if imagenet_norm and augment != -1:
        image_transforms.append(transforms.Normalize(Constants.IMAGENET_MEAN, Constants.IMAGENET_STD))             
    
    datasets = []
    for e in envs:
        func = preprocess.get_process_func(e)
        paths = Constants.df_paths[e]
        
        if split is not None:    
            splits = [split]
        else:
            splits = ['train', 'val', 'test']
            
        if augmented_dfs is not None: # use provided dataframes for subsample augmentation
            dfs = [augmented_dfs[e][i] for i in splits]
        else:            
            dfs = [func(pd.read_csv(paths[i]), only_frontal) for i in splits]            
            
        for c, s in enumerate(splits):
            cache_dir = Path(Constants.cache_dir)/ f'{e}_{s}/'
            cache_dir.mkdir(parents=True, exist_ok=True)
            datasets.append(AllDatasetsShared(dfs[c], transform = transforms.Compose(image_transforms)
                                      , split = split, cache = cache, cache_dir = cache_dir, subset_label = subset_label, output_type = output_type,
                                      ifft_filter = ifft_filter, add_noise = add_noise, pixel_thres = pixel_thres, crop_patch_at_end = crop_patch_at_end,
                                      patched = patched, patch_ind = patch_ind)) 
                
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = ConcatDataset(datasets)
        ds.dataframe = pd.concat([i.dataframe for i in datasets])
    
    return ds

class AllDatasetsShared(Dataset):
    def __init__(self, dataframe, transform=None, split = None, cache = True, cache_dir = '', subset_label = None, output_type = 'normal', ifft_filter = None,
                    add_noise = False, pixel_thres = None, crop_patch_at_end = False, patched = 'none', patch_ind = None):
        super().__init__()
        self.dataframe = dataframe
        self.dataset_size = self.dataframe.shape[0]
        self.transform = transform
        self.split = split
        self.cache = cache
        self.cache_dir = Path(cache_dir)
        self.subset_label = subset_label # (str) select one label instead of returning all Constants.take_labels
        self.output_type = output_type
        if ifft_filter is not None and ifft_filter[0] in ['low', 'high']:
            self.ifft_filter = ifft_filter[:2]
        else:
            self.ifft_filter = ifft_filter
        self.add_noise = add_noise
        self.pixel_thres = pixel_thres
        self.crop_patch_at_end = crop_patch_at_end
        self.patched = patched
        self.patch_ind = patch_ind

        if self.output_type == 'ifft':
            if self.ifft_filter[0] in ['low','high']:
                self.filter = draw_circle(shape = (224, 224), diameter = self.ifft_filter[1])
                if self.ifft_filter[0] == 'high':
                    self.filter = ~self.filter
            elif self.ifft_filter[0] in ['bandpass', 'notch']:
                filter_outer = draw_circle(shape = (224, 224), diameter = self.ifft_filter[2])
                filter_inner = draw_circle(shape = (224, 224), diameter = self.ifft_filter[1])
                if self.ifft_filter[0] == 'notch':
                    self.filter = ~(~filter_inner & filter_outer)
                else:                    
                    self.filter = (~filter_inner & filter_outer)                
            else:
                raise NotImplementedError

    def get_cache_path(self, cache_dir, meta):
        path = Path(meta['path'])
        if meta['env'] in ['PAD', 'NIH']:
            return cache_dir / (path.stem + '.pkl')
        elif meta['env'] in ['MIMIC', 'CXP']:
            return (cache_dir / '_'.join(path.parts[-3:])).with_suffix('.pkl')  
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        cache_path = self.get_cache_path(self.cache_dir, item)
        
        if self.cache and cache_path.is_file():
            img, label, meta = pickle.load(cache_path.open('rb'))
            meta = item.to_dict() # override
        else:            
            img = np.array(Image.open(item["path"]))

            if img.dtype == 'int32':
                img = np.uint8(img/(2**16)*255)
            elif img.dtype == 'bool':
                img = np.uint8(img)
            else: #uint8
                pass

            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)            
            elif len(img.shape)>2:
                # print('different shape', img.shape, item)
                img = img[:,:,0]
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2) 

            img = Image.fromarray(img)
            resize_transform = transforms.Resize(size = [224, 224])            
            img = transforms.Compose([resize_transform])(img)            

            label = torch.FloatTensor(np.zeros(len(Constants.take_labels), dtype=float))
            for i in range(0, len(Constants.take_labels)):
                if (self.dataframe[Constants.take_labels[i].strip()].iloc[idx].astype('float') > 0):
                    label[i] = self.dataframe[Constants.take_labels[i].strip()].iloc[idx].astype('float')

            meta = item.to_dict()
            
            if self.cache:
                pickle.dump((img, label, meta), cache_path.open('wb'))
        
        if self.transform is not None: # apply image augmentations after caching
            img = self.transform(img)
        
        if self.subset_label:
            if self.subset_label in Constants.take_labels:
                label = int(label[Constants.take_labels.index(self.subset_label)])
            elif self.subset_label == 'gender':
                label = meta['Sex']
            elif self.subset_label == 'race':
                label = meta['race']              
            elif self.subset_label == 'insurance':
                label = meta['insurance']
            else:
                raise NotImplementedError

        if self.add_noise: # add gaussian noise
            epsilon = 0.25 
            _, img = random_attack(img, epsilon)

        if self.output_type == 'fft':
            img = img[0, :, :].float().numpy()
            spectra = fft(img)
            img = torch.from_numpy(np.stack([spectra, spectra, spectra]))

        elif self.output_type == 'ifft':     
            img = img[0, :, :].float().numpy()
            spectra = np.fft.fftshift(np.fft.fft2(img))
            img = filter_and_ifft(spectra, self.filter)
            img = torch.from_numpy(np.stack([img, img, img]))
        else:
            assert self.output_type == 'normal'

        if self.patched != 'none' and self.patched is not None:
            assert(not self.crop_patch_at_end)
            # assert(0 <= self.patch_ind < Constants.N_PATCHES)
            if self.patched == 'patch':
                img = split_tensor(img, tile_size=int(224/math.sqrt(Constants.N_PATCHES)), offset=int(224/math.sqrt(Constants.N_PATCHES)))[0][self.patch_ind]
                img = transforms.Resize((224, 224))(img)
            elif self.patched == 'invpatch':
                img = blacken_tensor(img, self.patch_ind, tile_size=int(224/math.sqrt(Constants.N_PATCHES)), offset=int(224/math.sqrt(Constants.N_PATCHES)))
            else:
                raise NotImplementedError

        if self.crop_patch_at_end:            
            img = transforms.RandomResizedCrop(size = 224, scale = (0.2, 0.2), ratio = (1.0, 1.0))(img)

        if self.pixel_thres is not None:
            img[img > self.pixel_thres] = self.pixel_thres
            
        return img, label, meta
            
    def __len__(self):
        return self.dataset_size
