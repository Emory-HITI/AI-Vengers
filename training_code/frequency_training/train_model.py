import numpy as np
import argparse
import sys  
import pickle 
import os 
import random 
from pathlib import Path
import math
import json

from lib import models
from lib.data import get_dataset
from lib.train_helper import prediction, train_step
from lib.utils import EarlyStopping, save_checkpoint, has_checkpoint, load_checkpoint
from lib.infinite_loader import InfiniteDataLoader
from sklearn.metrics import roc_auc_score
import Constants

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required = True)
parser.add_argument('--domain', type=str, choices = ['MIMIC', 'CXP', 'both'])
parser.add_argument('--target', type=str, choices = ['race', 'gender', 'Pneumonia'])
parser.add_argument('--es_patience', type=int, default=7) # *val_freq steps
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--model', type = str, choices = ['densenet', 'vision_transformer'], default = 'densenet')
parser.add_argument('--data_type', type = str, choices = ['normal','fft','ifft'], 
    help = '`normal`: train on orginal images, `fft`: train on frequency spectra (not used), `ifft` train on inverse transformed images')
parser.add_argument('--patched', type = str, choices = ['patch', 'invpatch', 'none'], default = 'none',
    help = '`none`: train on the whole image, `patch`: train using only patch patch_ind, `invpatch`: set patch_ind to black, train on whole image')
parser.add_argument('--patch_ind', type = int, choices = list(range(Constants.N_PATCHES)), default = None)
parser.add_argument('--filter_type', type = str, choices = ['low', 'high', 'notch', 'bandpass'])
parser.add_argument('--filter_thres', type = float)
parser.add_argument('--filter_thres2', type = float)
parser.add_argument('--augment', action = 'store_true', help = 'whether to use data augmentation')
parser.add_argument('--use_pretrained', action = 'store_true')
parser.add_argument('--add_noise', action = 'store_true')
parser.add_argument('--pixel_thres', type = float, default = 1.0, help = 'intensity threshold for clipping pixels, 1.0 for no clipping')
parser.add_argument('--crop_patch_at_end', action = 'store_true')
parser.add_argument('--debug', action = 'store_true')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True  

output_dir = Path(args.output_dir)
output_dir.mkdir(parents = True, exist_ok = True)

with open(Path(output_dir)/'args.json', 'w') as outfile:
    json.dump(vars(args), outfile)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.target == 'race':
    classes = [Constants.race_mapping[i] for i in range(len(Constants.race_mapping))]    
elif args.target == 'gender':
    classes = [Constants.gender_mapping[i] for i in range(len(Constants.gender_mapping))]
else:
    classes = ['neg', 'pos']

n_outputs = len(classes)

if args.model == 'densenet':
    model = models.DenseNet(args.use_pretrained, n_outputs = n_outputs).to(device)
elif args.model == 'vision_transformer':
    model = models.VisionTransformer(args.use_pretrained, n_outputs = n_outputs).to(device)

print("Total parameters: " + str(sum(p.numel() for p in model.parameters())))

optimizer = optim.Adam(model.parameters(), lr = args.lr)   
lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=5, factor=0.1)

if args.domain in ['MIMIC', 'CXP']:
    envs = [args.domain]
elif args.domain == 'both':
    envs = ['MIMIC', 'CXP']
else:
    raise NotImplementedError

common_args = {
    'envs': envs,
    'only_frontal': True,
    'subset_label': args.target,
    'output_type':  args.data_type,
    'imagenet_norm': not args.data_type == 'fft',
    'augment': int(args.augment),
    'ifft_filter': (args.filter_type, args.filter_thres, args.filter_thres2),
    'add_noise': args.add_noise,
    'pixel_thres': args.pixel_thres,
    'crop_patch_at_end': args.crop_patch_at_end,
    'patched': args.patched,
    'patch_ind': args.patch_ind
}

train_data = get_dataset(split = 'train', **common_args)
val_data = get_dataset(split = 'val', **common_args)
test_data = get_dataset(split = 'test', **common_args)
test_data_aug = get_dataset(augment = 1, split = 'test', **{i:common_args[i] for i in common_args if i != 'augment'})

if args.debug:
    val_data = Subset(val_data, list(range(1024)))
    test_data = Subset(test_data, list(range(1024)))
    test_data_aug = Subset(test_data_aug, list(range(1024)))
else:
    val_data = Subset(val_data,  np.random.choice(np.arange(len(val_data)), min(1024*8, len(val_data)), replace = False))

es = EarlyStopping(patience = args.es_patience)  
batch_size = args.batch_size
if args.debug:
    n_steps = 50
else:
    n_steps = args.epochs * (len(train_data) // batch_size) 

train_loader = InfiniteDataLoader(train_data, batch_size=batch_size, num_workers = 1)
validation_loader = DataLoader(val_data, batch_size=batch_size*2, shuffle=False) 
test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=False) 
test_loader_aug = DataLoader(test_data_aug, batch_size=batch_size*2, shuffle=False) 

if has_checkpoint() and not args.debug:
    state = load_checkpoint()
    model.load_state_dict(state['model_dict'])
    optimizer.load_state_dict(state['optimizer_dict'])
    lr_scheduler.load_state_dict(state['scheduler_dict'])
    train_loader.sampler.load_state_dict(state['sampler_dict'])
    start_step = state['start_step']
    es = state['es']
    torch.random.set_rng_state(state['rng'])
    print("Loaded checkpoint at step %s" % start_step)
else:
    start_step = 0  

for step in range(start_step, n_steps):    
    if es.early_stop:
        break               
    data, target, meta = next(iter(train_loader))
    step_loss, step_acc = train_step(data, target, model, 
                                     device, optimizer) 

    print('Train Step: {} Accuracy: {:.4f}\tLoss: {:.6f}'.format(
                step, step_acc, step_loss), flush = True)

    if step % args.val_freq == 0:
        total_loss, prob_list, pred_list, target_list = prediction(model, device, validation_loader)
        lr_scheduler.step(total_loss)
        es(total_loss, step , model.state_dict(), output_dir/'model.pt')  

        save_checkpoint(model, optimizer, lr_scheduler,
                            train_loader.sampler.state_dict(train_loader._infinite_iterator), 
                            step+1, es, torch.random.get_rng_state())


_, prob_list, _, target_list = prediction(model, device, test_loader)
_, prob_list_aug, _, target_list_aug = prediction(model, device, test_loader_aug)

results = {}
for m, prob, target in zip(['unaug', 'aug'], [prob_list, prob_list_aug], [target_list, target_list_aug]):
    results[m] = {}
    for grp in np.unique(target):
        target_bin = target == grp
        pred = prob[:, grp]
        results[m][f'roc_{classes[grp]}'] = roc_auc_score(target_bin, pred)

    results[m]['preds'] = prob
    results[m]['targets'] = target

if args.debug:
    print(results)

pickle.dump(results, (output_dir/'results.pkl').open('wb'))

with open(output_dir/'done', 'w') as f:
    f.write('done')    
