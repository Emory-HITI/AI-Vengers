import numpy as np
import argparse
import sys
import pickle
import os
import sys
import random
from pathlib import Path
import math
import json

from lib import models
from lib.data import get_dataset
from lib.perturb_helper import fgsm_attack, random_attack
from sklearn.metrics import roc_auc_score, accuracy_score
import Constants

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required = True)
parser.add_argument('--domain', type=str, choices = ['MIMIC', 'CXP', 'both'])
parser.add_argument('--plabel', type=str, choices = ['race', 'gender', 'Pneumonia'])
parser.add_argument('--model', type = str, choices = ['densenet', 'vision_transformer'], default = 'densenet')
parser.add_argument('--use_pretrained', action = 'store_true')
parser.add_argument('--input_dir', type = str, required = True)
parser.add_argument('--attack', type = str, choices = ['fgsm', 'random'], default = 'fgsm')
parser.add_argument('--epsilons', nargs='+', type = float, default = np.linspace(0., 0.5, 10).tolist())
parser.add_argument('--data_type', type = str, choices = ['normal'])
parser.add_argument('--patched', type = str, choices = ['patch', 'invpatch', 'none'], default = 'none')
parser.add_argument('--patch_ind', type = int, choices = list(range(Constants.N_PATCHES)), default = None)
parser.add_argument('--augment', action = 'store_true')
parser.add_argument('--pixel_thres', type = float, default = 1.0)
parser.add_argument('--debug', action = 'store_true')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_dir', type=str, required=True)
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

config = {}
for indir in os.listdir(Path(args.input_dir)):
    model_path = os.path.join(*[args.input_dir, indir, 'model.pt'])
    args_path = os.path.join(*[args.input_dir, indir, 'args.json'])
    if os.path.exists(model_path) and os.path.exists(args_path):
        with open(args_path, 'r') as fp:
            try:
                train_args = json.load(fp)
            except:
                print('Unexpected error encountered trying to load JSON {}: {}'.format(args_path, sys.exc_info()[0]))
                train_args = {}
        use = True
        for key in train_args:
            if key in vars(args) and key not in ['exp_name', 'output_dir', 'seed', 'debug', 'batch_size'] and vars(args)[key] != train_args[key]:
                use = False
                break
        if use and 'target' in train_args:
            config[train_args['target']] = model_path

if args.debug:
    assert args.plabel in config, "Label with which to perturb model must exist as key in the config."
    
models_dict = {}
classes_dict = {}
for key in config:
    
    if key == 'race':
        classes_dict[key] = [Constants.race_mapping[i] for i in range(len(Constants.race_mapping))]
    elif key == 'gender':
        classes_dict[key] = [Constants.gender_mapping[i] for i in range(len(Constants.gender_mapping))]
    else:
        classes_dict[key] = ['neg', 'pos']

    n_outputs = len(classes_dict[key])

    if args.model == 'densenet':
        model = models.DenseNet(args.use_pretrained, n_outputs = n_outputs).to(device)
    elif args.model == 'vision_transformer':
        raise NotImplementedError("Vision transformer not currently supported on this branch.")
        
    checkpoint_state_dict = torch.load(config[key])
    model.load_state_dict(checkpoint_state_dict)
    
    model.eval()
    models_dict[key] = model
    
    print(f"Key {key}: Total parameters: " + str(sum(p.numel() for p in model.parameters())))

if args.domain in ['MIMIC', 'CXP']:
    envs = [args.domain]
elif args.domain == 'both':
    envs = ['MIMIC', 'CXP']
else:
    raise NotImplementedError

common_args = {
    'envs': envs,
    'only_frontal': True,
    'subset_label': args.plabel,
    'output_type':  args.data_type,
    'imagenet_norm': not args.data_type == 'fft',
    'augment': int(args.augment),
    'patched': args.patched,
    'patch_ind': args.patch_ind
}

test_data = get_dataset(split = 'test', **common_args)
test_data_aug = get_dataset(augment = 1, split = 'test', **{i:common_args[i] for i in common_args if i != 'augment'})

if args.debug:
    test_data = Subset(test_data, list(range(1024)))
    test_data_aug = Subset(test_data_aug, list(range(1024)))

batch_size = args.batch_size

test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
test_loader_aug = DataLoader(test_data_aug, batch_size=1, shuffle=False)

if args.attack == 'fgsm':
    attack_func = fgsm_attack
elif args.attack == 'random':
    attack_func = random_attack
else:
    raise ValueError("Attack must be either 'fgsm' or 'random.'")

auc_dict = {}
acc_dict = {}

for epsilon in args.epsilons:

    targets_dict = {key: [] for key in config}
    preds_dict = {key: [] for key in config}
    probs_dict = {key: [] for key in config}

    # Loop over all examples in test set
    for data, target, meta in test_loader:

        data, target = data.float().to(device), target.long().to(device)
        target = target.view((-1))
        
        data.requires_grad = True
        
        output = models_dict[args.plabel](data)

        init_probs = F.softmax(output, dim=1)
        init_pred = init_probs.max(1, keepdim=True)[1]

        targets_dict[args.plabel].append(target.cpu().item())
        for key in config:
            if key != args.plabel:
                if key == 'gender':
                    targets_dict[key].append(meta['Sex'].item())
                    continue
                targets_dict[key].append(meta[key].item())
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item() or epsilon == 0.:
            probs_dict[args.plabel].append(init_probs.squeeze().detach().cpu().numpy())
            preds_dict[args.plabel].append(init_pred.cpu().item())
            for key in config:
                if key != args.plabel:
                    with torch.no_grad():
                        output = models_dict[key](data)
                    
                    probs = F.softmax(output, dim=1)
                    preds = probs.max(1, keepdim=True)[1]
                    
                    probs_dict[key].append(probs.squeeze().cpu().numpy())
                    preds_dict[key].append(preds.cpu().item())
            continue

        # Get the loss
        loss = F.cross_entropy(output, target)
        
        # Zero all existing gradients
        models_dict[args.plabel].zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data
        
        attack_dict = {'batch': data, 'epsilon': epsilon}
        if args.attack == 'fgsm':
            attack_dict['data_grads'] = data_grad
            
        # Call Attack
        perturbed_data = attack_func(**attack_dict)

        # Re-classify the perturbed image
        for key in config:
            with torch.no_grad():
                perturbed_output = models_dict[key](perturbed_data)
            
            perturbed_probs = F.softmax(perturbed_output, dim=1)
            perturbed_pred = perturbed_probs.max(1, keepdim=True)[1]
            
            probs_dict[key].append(perturbed_probs.squeeze().cpu().numpy())
            preds_dict[key].append(perturbed_pred.item())
    
    auc_dict[epsilon] = {}
    for key in config:
        prob = np.stack(probs_dict[key])
        target = np.hstack(targets_dict[key])
        auc_dict[epsilon][key] = {} 
        for grp in np.unique(target):
            target_bin = target == grp
            pred = prob[:, int(grp)]
            auc_dict[epsilon][key][f'roc_{classes_dict[key][int(grp)]}'] = roc_auc_score(target_bin, pred)
    acc_dict[epsilon] = {key: accuracy_score(targets_dict[key], preds_dict[key]) for key in config}

#### RESULTS

results = {'auc': auc_dict, 'acc': acc_dict}
print(results)

with open(output_dir/'results.json', 'w') as fp:
    json.dump(results, fp)

with open(output_dir/'done', 'w') as f:
    f.write('done')
