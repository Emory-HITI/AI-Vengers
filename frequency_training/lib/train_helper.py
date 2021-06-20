import torch
import torch.nn.functional as F
import numpy as np

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
