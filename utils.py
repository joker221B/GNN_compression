import numpy as np
import torch
from sklearn.metrics import f1_score


def evaluate(features, model, subgraph, labels, loss_fcn):
    model.eval()
    loss_data, score = runnograd(model, subgraph, features, labels, loss_fcn)
    model.train()
    return score, loss_data.item()


def test_model(test_dataloader, model, device, loss_fcn):
    test_score_list = []
    model.eval()
    with torch.no_grad():
        for batch, test_data in enumerate(test_dataloader):
            subgraph, feats, labels = test_data
            labels = labels.to(device)
            feats = feats.to(device)
            score, val_loss = evaluate(feats, model, subgraph, labels.float(), loss_fcn)
            test_score_list.append(score)
        avgscore = np.array(test_score_list).mean()
        print(f"Testset Score F-1:        {avgscore:.4f}")
    model.train()
    return avgscore


def runnograd(model, subgraph, features, labels, loss_fcn):
    with torch.no_grad():
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(features.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average='micro')
        return loss_data, score


def evaluate_model(valid_dataloader, device, model, loss_fcn):
    val_loss_list = []
    score_list = []
    model.eval()
    with torch.no_grad():
        for batch, valid_data in enumerate(valid_dataloader):
            subgraph, feats, labels = valid_data
            labels = labels.to(device)
            feats = feats.to(device)
            score, val_loss = evaluate(feats.float(), model, subgraph, labels.float(), loss_fcn)
            val_loss_list.append(val_loss)
            val_loss = 0
            score_list.append(score)
            score= 0
    mean_score = np.array(score_list).mean()
    mean_val_loss = np.array(val_loss_list).mean()
    print(f"Valset Score F1 :        {mean_score:.4f} ")
    model.train()
    return mean_score

