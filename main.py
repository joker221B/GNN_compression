from copy import deepcopy
import numpy as np
import dgl
import dgl.function as fn
from torch.utils.data import DataLoader
import torch.nn.functional as F


from dgl.nn.pytorch.softmax import edge_softmax
from dgl.data.ppi import LegacyPPIDataset as PPIDataset
import torch
import torch.nn as nn
import argparse
from mutual_loss import optimizing, mutual_loss
from utils import evaluate,evaluate_model, test_model
import time
from dgl.nn.pytorch import GATConv


torch.set_num_threads(1)


class GAT(nn.Module):
    def __init__(self,g,layers,in_dim,hidden_layers,classes,head,activation,dropfeatures,drop_attention,slope,res):
        super(GAT, self).__init__()
        self.activation = activation
        self.gat_layers = nn.ModuleList()
        self.layers = layers
        self.g = g
        self.gat_layers.append(GATConv(in_dim, hidden_layers, head[0],dropfeatures, drop_attention, slope, False, None))
        for l in range(1, layers):
            self.gat_layers.append(GATConv(hidden_layers * head[l-1], hidden_layers, head[l],dropfeatures, drop_attention, slope, res, None))
        self.gat_layers.append(GATConv(hidden_layers * head[-2], classes, head[-1],dropfeatures, drop_attention, slope, res, None))

    def forward(self, inputs, flag=False):
        x = inputs
        features = []
        for l in range(self.layers):
            x = self.gat_layers[l](self.g, x).flatten(1)
            features.append(x)
            x = self.activation(x)
        logits = self.gat_layers[-1](self.g, x).mean(1)
        if flag:
            return logits, features
        return logits





class distance(nn.Module):
    def __init__(self):
        super(distance, self).__init__()
        
    def forward(self, graph, feats):
        graph = graph.local_var()
        feats = feats.view(-1, 1, feats.shape[1])
        graph.ndata.update({'ftl': feats, 'ftr': feats})



        graph.apply_edges(fn.u_dot_v('ftl', 'ftr', 'dotp'))
        edge_data = graph.edata.pop('dotp')
        edge_data = torch.exp( (-1.0/100) * torch.sum(torch.abs(edge_data), dim=-1) )
        edge_data = edge_softmax(graph, edge_data)
        return edge_data


def local_distance_model(feat_info, upsampling=False):
    return distance()

'''
def parameters(model):
    num_params = 0
    for params in model.parameters():
        curn = 1
        for size in params.data.shape:
            curn *= size
        num_params += curn
        print()
    return num_params
'''

def combine(element):
    graphs, feats, labels =map(list, zip(*element))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels

def optimize(optimizer,loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_teacher(args, model, data, device):
    res_model = None
    max_val = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0)
    ce_loss = torch.nn.BCEWithLogitsLoss()
    train_dataloader, valid_dataloader, test_dataloader, _ = data
    for epoch in range(args.epochs_t):
        
        losses_list = []
        model.train()
        for batch, b_data in enumerate(train_dataloader):
            
            subgraph, feas, orig_labels = b_data
            for layer in model.gat_layers:
                layer.g = subgraph
            model.g = subgraph
            orig_labels = orig_labels.to(device)
            feas = feas.to(device)
            output = model(feas.float())
            loss = ce_loss(output, orig_labels.float())
            losses_list.append(loss.item())
            optimize(optimizer,loss)
            
            
        curr_loss = np.array(losses_list).mean()
        
        print(f"Current Epoch {epoch :04d} | Current Loss: {curr_loss:.3f}")
        if epoch % 20 == 0:

            val_loss_l = []
            score_l = []
            
            for batch, v_data in enumerate(valid_dataloader):
                subgraph, feas, orig_labels = v_data
                feas = feas.to(device)
                orig_labels = orig_labels.to(device)
                s, v = evaluate(feas.float(), model, subgraph, orig_labels.float(), ce_loss)
                
                val_loss_l.append(v)
                score_l.append(s)
                
            curr_mean = np.array(score_l).mean()
            mean_val_loss = np.array(val_loss_l).mean()
            

            train_score_l = []
            if  curr_mean > max_val:
                res_model = deepcopy(model)
            for batch, train_data in enumerate(train_dataloader):
                subgraph, feas, orig_labels = train_data

                orig_labels = orig_labels.to(device)
                feas = feas.to(device)
                
                train_score_l.append(evaluate(feas, model, subgraph, orig_labels.float(), ce_loss)[0])
            print(f"TrainSet F-1:        {np.array(train_score_l).mean():.4f}")



    test_score_l = []
    for batch, t_data in enumerate(test_dataloader):
        subgraph, feas, orig_labels = t_data

        orig_labels = orig_labels.to(device)
        feas = feas.to(device)

        test_score_l.append(evaluate(feas, model, subgraph, orig_labels.float(), ce_loss)[0])
    print(f"TestSet F-1:        {np.array(test_score_l).mean():.4f}")


def update_layers(stud_model,subgraph):
    for layer in stud_model.gat_layers:
        layer.g = subgraph
    return

def train_student(args, custom_model, data, device):
    min_loss = 100000.0
    max_score = 0
    ce_loss = torch.nn.BCEWithLogitsLoss() #cross entropy for multiclassification
    mse_loss = torch.nn.MSELoss()
    train_dataloader, valid_dataloader, test_dataloader, fixed_dataloader = data
    stud_model = custom_model['stud_model']['model']
    teach_model = custom_model['teach_model']['model']
    step_n = 0
    for epoch in range(args.epochs_s):
        
        losses_list = []
        additional_losses_list = []
        stud_model.train()
        start = time.time()

        for batch, batch_data in enumerate( zip(train_dataloader,fixed_dataloader) ):
            
            data_wshuffle, _ = batch_data
            subgraph, feas, orig_labels = data_wshuffle
            stud_model.g = subgraph
            update_layers(stud_model,subgraph)
            feas = feas.to(device)
            orig_labels = orig_labels.to(device)
            logits, middle_feats_s = stud_model(feas.float(), flag=True)
            step_n += 1
            if epoch >= args.complete:
                args.mode = 'complete'

            if args.mode != 'complete':
                teach_model.eval()
                with torch.no_grad():
                    teach_model.g = subgraph
                    for layer in teach_model.gat_layers:
                        layer.g = subgraph

                    logits_t = teach_model(feas.float())
                    logits_t.detach()


                
                transfer_loss = mutual_loss(custom_model, middle_feats_s[args.target_layer], subgraph, feas)
                additional_loss = (args.loss_weight * transfer_loss)
                loss = additional_loss + ce_loss(logits, orig_labels.float()) 

            else:
                loss = ce_loss(logits, orig_labels.float())
                additional_loss = torch.tensor(0).to(device)

            optimizing(custom_model, loss, ['stud_model'])
            losses_list.append(loss.item())
            additional_losses_list.append(additional_loss.item() if additional_loss!=0 else 0)

        curr_loss = np.array(losses_list).mean()
        curr_additional = np.array(additional_losses_list).mean()
        print(f"Epoch {epoch:04d} | CELoss: {curr_loss:.5f} | Mutual : {curr_additional:.5f} | Time: {time.time()-start:.2f}s")
        if epoch % 20 == 0:
            curr_s = evaluate_model(valid_dataloader, device, stud_model, ce_loss)
            if curr_s > max_score or curr_loss < min_loss:
                max_score = curr_s
                min_loss = curr_loss
                test_score = test_model(test_dataloader, stud_model, device, ce_loss)
    print(f"TestSet F-1: {test_score:.4f}")



    
def get_feature_info(args):
    feature_info = {}
    feature_info['stud_feat'] = [args.num_heads_s*args.num_hidden_s] * args.num_layers_s
    feature_info['teach_feat'] = [args.t_num_heads*args.t_num_hidden] * args.t_num_layers
    return feature_info

def main(args):
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=combine, num_workers=4, shuffle=True)
    fixed_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=combine, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=combine, num_workers=3)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=combine, num_workers=3)
    data = (train_dataloader, valid_dataloader, test_dataloader, fixed_dataloader)
    graph = train_dataset.graph
    num_feats = train_dataset.features.shape[1]
    num_classes = train_dataset.labels.shape[1]
    data_info = {}
    data_info['num_classes'] = num_classes
    data_info['num_feats'] = num_feats
    data_info['graph'] = graph
    device = torch.device("cpu")
    feature_info = get_feature_info(args)
    local_dist_model = local_distance_model(feature_info)
    local_dist_model.to(device)
    local_dist_model_optimizer = None
    stud_heads = ([args.num_heads_s] * args.num_layers_s) + [args.num_out_heads_s]
    stud_model = GAT(data_info['graph'],args.num_layers_s,data_info['num_feats'],args.num_hidden_s,data_info['num_classes'],stud_heads,F.elu,0,0,args.alpha,True)
    stud_model.to(device)
    stud_model_optimizer = torch.optim.SGD(stud_model.parameters(), lr=args.lr, weight_decay=0)
    
    teach_heads = ([args.t_num_heads] * args.t_num_layers) + [args.t_num_out_heads]
    teach_model = GAT(data_info['graph'], args.t_num_layers, data_info['num_feats'], args.t_num_hidden, data_info['num_classes'],teach_heads,F.elu,0,0,args.alpha,True)                        
    teach_model.to(device)
    teach_model_optimizer = torch.optim.SGD(teach_model.parameters(), lr=args.lr, weight_decay=0)
    model_dict = {}
    model_dict['stud_model'] = {'model':stud_model, 'optimizer':stud_model_optimizer}
    model_dict['local_dist_model'] = {'model':local_dist_model, 'optimizer':local_dist_model_optimizer}
    model_dict['teach_model'] = {'model':teach_model, 'optimizer':teach_model_optimizer}

    teach_model = model_dict['teach_model']['model']
    stud_model = model_dict['stud_model']['model']
    print("####### Teacher GAT start ########")
    #train_teacher(args, teach_model, data, device)
    train_dataloader, _, test_dataloader, _ = data
    ce_loss = torch.nn.BCEWithLogitsLoss()
    print(f"accuracy for teacher GAT on trainset:")
    test_model(train_dataloader, teach_model, device, ce_loss)
    print(f"accuracy for teacher GAT on testset:")
    test_model(test_dataloader, teach_model, device, ce_loss)
    print("#### Student GAT start ######")
    train_student(args, model_dict, data, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Attention')
    parser.add_argument("--num-layers-s", type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)    
    parser.add_argument('--loss-weight', type=float, default=1.0)
    parser.add_argument("--num-heads-s", type=int, default=2)
    parser.add_argument("--t-num-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-out-heads-s", type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument("--epochs-s", type=int, default=500)
    parser.add_argument("--target-layer", type=int, default=2)
    parser.add_argument("--mode", type=str, default='mi')
    parser.add_argument('--complete', type=int, default=50)
    parser.add_argument("--t-num-layers", type=int, default=2)
    parser.add_argument("--epochs-t", type=int, default=70)
    parser.add_argument("--t-num-out-heads", type=int, default=6)
    parser.add_argument("--t-num-hidden", type=int, default=256)
    parser.add_argument("--num-hidden-s", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(100)
    print(args)
    main(args)