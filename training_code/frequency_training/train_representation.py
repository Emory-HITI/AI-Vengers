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
from lib.train_helper import representation, train_triplet_step
from lib.utils import EarlyStopping, save_checkpoint, has_checkpoint, load_checkpoint
from lib.infinite_loader import InfiniteDataLoader
import Constants

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset

from pytorch_metric_learning import miners
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required = True)
parser.add_argument('--domain', type=str, choices = ['MIMIC', 'CXP', 'both'])
parser.add_argument('--target', type=str, choices = ['race', 'gender', 'Pneumonia'])
parser.add_argument('--es_patience', type=int, default=7) # *val_freq steps
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--model', type = str, choices = ['densenet', 'vision_transformer'], default = 'densenet')
parser.add_argument('--embed_dim', type = int, default=128)
parser.add_argument('--data_type', type = str, choices = ['normal','fft','ifft'])
parser.add_argument('--augment', action = 'store_true')
parser.add_argument('--use_pretrained', action = 'store_true')
parser.add_argument('--pixel_thres', type = float, default = 1.0)
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

n_outputs = args.embed_dim

if args.model == 'densenet':
    model = models.DenseNet(args.use_pretrained, n_outputs = n_outputs).to(device)
elif args.model == 'vision_transformer':
    raise NotImplementedError("Vision transformer not currently supported on this branch.")

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

miner = miners.DistanceWeightedMiner()

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
    step_loss, step_precision = train_triplet_step(data, target, model,
                                     device, optimizer, miner)

    print('Train Step: {} Precision@1: {:.4f}\tLoss: {:.6f}'.format(
                step, step_precision, step_loss), flush = True)

    if step % args.val_freq == 0:
        total_loss, acc_dict, embedding_list, target_list = representation(model, device, validation_loader)
        lr_scheduler.step(total_loss)
        es(total_loss, step , model.state_dict(), output_dir/'model.pt')

        save_checkpoint(model, optimizer, lr_scheduler,
                            train_loader.sampler.state_dict(train_loader._infinite_iterator),
                            step+1, es, torch.random.get_rng_state())


_, acc_dict, embedding_list, target_list = representation(model, device, test_loader)
_, acc_dict_aug, embedding_list_aug, target_list_aug = representation(model, device, test_loader_aug)

results = {}
acc_calc = AccuracyCalculator()
for m, embedding, target in zip(['unaug', 'aug'], [embedding_list, embedding_list_aug], [target_list, target_list_aug]):
    results[m] = {}
    for grp in np.unique(target):
        target_bin = target == grp
        embedding_bin = embedding[target_bin,:]
        results[m][f'metrics_{classes[grp]}'] = acc_calc.get_accuracy(embedding_bin, embedding_bin, target_bin, target_bin, embeddings_come_from_same_source=True)

    results[m]['targets'] = target

if args.debug:
    print(results)

pickle.dump(results, (output_dir/'results.pkl').open('wb'))

with open(output_dir/'done', 'w') as f:
    f.write('done')

