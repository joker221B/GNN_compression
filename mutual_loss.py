import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dgl.nn.pytorch.softmax import edge_softmax
import dgl
import dgl.function as fn


def dozero(custom_model, model_list):
    for model in model_list:
        custom_model[model]['optimizer'].zero_grad()

def optimizing(custom_model, loss, model_list):
    dozero(custom_model, model_list)
    loss.backward()
    dostep(custom_model, model_list)


def runnograd(teach_model, subgraph, feats):
    with torch.no_grad():
        teach_model.g = subgraph
        for layer in teach_model.gat_layers:
            layer.g = subgraph
        _, middle_feats_t = teach_model(feats.float(), flag=True)
        middle_feats_t = middle_feats_t[1]
        return middle_feats_t


def KL_loss(edgex, edgey,graph):
    with graph.local_scope():
        nnode = graph.number_of_nodes()
        graph.ndata.update({'kldiv': torch.ones(nnode,1).to(edgex.device)})
        temp = torch.log(edgey)-torch.log(edgex)
        diff = edgey*temp
        graph.edata.update({'diff':diff})
        graph.update_all(fn.u_mul_e('kldiv', 'diff', 'm'),fn.sum('m', 'kldiv'))
        flat=torch.flatten(graph.ndata['kldiv'])
        return torch.mean(flat)
    


    

def dostep(custom_model, model_list):
    for model in model_list:
        custom_model[model]['optimizer'].step()


def mutual_loss(custom_model, middle_feats_s, subgraph, feats):
    loss_fcn = nn.MSELoss(reduction="mean")
    teach_model = custom_model['teach_model']['model']

    middle_feats_t = runnograd(teach_model, subgraph, feats)
    dist_teach = custom_model['local_dist_model']['model'](subgraph, middle_feats_t)
    dist_stud = custom_model['local_dist_model']['model'](subgraph, middle_feats_s)
    return KL_loss(dist_stud, dist_teach,subgraph)






