import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

def train_step(data, target, model, device, optimizer, extras = None):
    model.train() 
    
    data = data.float().to(device) 
    target = target.long().to(device)
    target = target.view((-1))
    
    if torch.is_tensor(extras):
        extras = extras.float().to(device)

    output = model(data, extras) ## no softmax applied 

    loss = F.cross_entropy(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred = F.softmax(output, dim=1).max(1, keepdim=True)[1]

    acc = 1.*pred.eq(target.view_as(pred)).sum().item()/len(data)
        
    return loss.item(), acc


def prediction(model, device, data_loader):
    model.eval() 
    prob_list = []; pred_list = [];  target_list = []
    total_loss = 0.0
    with torch.no_grad():
        for [data, target, extras] in data_loader:
            data = data.float().to(device) # add channel dimension
            target = target.long().to(device)
            target = target.view((-1,))
            
            if torch.is_tensor(extras):
                extras = extras.float().to(device)
            
            target_list = target_list + list(target.cpu().detach().tolist())

            output = model(data, extras)   
            
            total_loss += F.cross_entropy(output, target) 
            
            prob = F.softmax(output, dim=1)
            prob_list = prob_list + list(prob.cpu().detach().tolist())
            
            pred = prob.max(1, keepdim=True)[1]
            pred_list = pred_list + list(pred.cpu().detach().tolist())

    total_loss /= len(data_loader.dataset)

    return total_loss, np.array(prob_list), np.array(pred_list), np.array(target_list)

def train_triplet_step(data, target, model, device, optimizer, miner, extras = None):
    model.train()
        
    loss_func = losses.MarginLoss()
    acc_calc = AccuracyCalculator()
    
    data = data.float().to(device)
    target = target.long().to(device)
    target = target.view((-1))
    
    if torch.is_tensor(extras):
        extras = extras.float().to(device)

    embedding = model(data, extras)
    triplets = miner.mine(embedding, target, embedding, target)
    
    loss = loss_func(embedding, target, triplets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        acc_dict = acc_calc.get_accuracy(embedding, embedding, target, target, embeddings_come_from_same_source=True)
    
    return loss.item(), acc_dict["precision_at_1"]

def representation(model, device, data_loader):
    model.eval()
    target_list = []
    embedding_list = []
    total_loss = 0.0
        
    loss_func = losses.MarginLoss()
    acc_calc = AccuracyCalculator()
    miner = miners.BatchEasyHardMiner(pos_strategy='all', neg_strategy='all')
    
    acc_dicts = defaultdict(list)
    with torch.no_grad():
        for [data, target, extras] in data_loader:
            data = data.float().to(device) # add channel dimension
            target = target.long().to(device)
            target = target.view((-1,))
            
            if torch.is_tensor(extras):
                extras = extras.float().to(device)
            
            target_list = target_list + list(target.cpu().detach().tolist())

            embedding = model(data, extras)
            triplets = miner.mine(embedding, target, embedding, target)
            
            embedding_list = embedding_list + list(embedding.cpu().detach().tolist())
            
            total_loss += loss_func(embedding, target, triplets)
            
            acc_dict = acc_calc.get_accuracy(embedding, embedding, target, target, embeddings_come_from_same_source=True)
            for key in acc_dict:
                acc_dicts[key].append(acc_dict[key])

    total_loss /= len(data_loader.dataset)
    
    avg_acc_dict = {key: np.mean(acc_dicts[key]) for key in acc_dicts}
    return total_loss, avg_acc_dict, np.array(embedding_list), np.array(target_list)
