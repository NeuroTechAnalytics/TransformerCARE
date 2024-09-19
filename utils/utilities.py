import os
import random
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve
from collections import defaultdict
import matplotlib.pyplot as plt
import torch

import sys
sys.path.append("..")
from config import *


class Reports():

    def __init__(self):
        self.reset()

    def reset(self):
        self.final_results = defaultdict(lambda: {})
        self.evaluation_results = defaultdict(lambda: [])
        self.evaluation_probs = []


def set_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



def extract_embeddings(dataloader, model, device):

    transformer_model = model.transformer_model
    for param in transformer_model.parameters():
        param.requires_grad = False

    data_embedding = defaultdict(lambda: [])

    for batch in dataloader:
        file_names, input =  batch[2], batch[0].to(device)
        with torch.no_grad():
            output = transformer_model(input).last_hidden_state

        embeddings = torch.mean(output, dim = 1)

        for i, name in enumerate(file_names):
            data_embedding[name.split('.')[0]].append(embeddings[i].squeeze().cpu())

    for key in data_embedding:
        embeddings = torch.stack(data_embedding[key], dim = 0)
        data_embedding[key] = torch.mean(embeddings, dim = 0)

    return data_embedding



def save_classification_reports (reports, status, predicted_probs, predicted_labels, ture_labels):
    # Compute AUC
    fpr, tpr, _ = roc_curve(ture_labels, np.array(predicted_probs)[:,1])
    auc_ = auc(fpr, tpr)
    reports.final_results[status]['auc'] = round(auc_ * 100, 2)
    # Compute Precision
    prec = precision_score(ture_labels, predicted_labels)
    reports.final_results[status]['prec'] = round(prec * 100, 2)
    # Compute Recall
    recall = recall_score(ture_labels, predicted_labels)
    reports.final_results[status]['recall'] = round(recall * 100, 2)
    # Compute F1-score
    f1 = f1_score(ture_labels, predicted_labels)
    reports.final_results[status]['f1'] = round(f1 * 100, 2)
    # Compute Accuracy
    acc = accuracy_score(ture_labels, predicted_labels)
    reports.final_results[status]['acc'] = round(acc * 100, 2)



def gather_eval_results(reports, ids, pred_probs, true_label ):
        for i, id in enumerate(ids):
            reports.evaluation_results[id].append(((np.argmax(pred_probs[i]), np.max(pred_probs[i]) ), true_label[i]))



def add_eval_probs(reports, status, pred_probs, pred_labels, true_labels, ids):
        new_probs = pd.DataFrame({'seed':[seed]*len(true_labels),
                               'status':[status]*len(true_labels),
                               'id':ids,
                               'true_labels': true_labels,
                               'pred_labels': pred_labels,
                               'pred_probs_label_1' : pred_probs})
        reports.evaluation_probs.append(new_probs)



def plot_training (loss_list, metric_list, labels, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 5))
    fig.subplots_adjust(wspace = .2)
    plotLoss(ax1, loss_list, labels, title)
    plotMetric(ax2, metric_list, labels, title)
    plt.show()
    # fig.savefig(result_path + eval_results_folder +'/'+ title)



def plotLoss (ax, loss_list, labels, title):
    if loss_list.all() == None : return
    for i, label in enumerate(labels):
        ax.plot(loss_list[:, i], label = label)
    ax.set_title("Loss Curves - " + title, fontsize = 9)
    ax.set_ylabel("Loss", fontsize = 8)
    ax.set_xlabel("Epoch", fontsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
    ax.legend(prop = {'size': 8})
    ax.grid()



def plotMetric (ax, metric_list, labels, title):
    if metric_list.all() == None : return
    if ax == None: fig, ax = plt.subplots()
    for i, label in enumerate(labels):
        ax.plot(metric_list[:, i], label = label)
    ax.set_title("Metric Curve - " + title, fontsize = 9)
    ax.set_ylabel("Score", fontsize = 8)
    ax.set_xlabel("Epoch", fontsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
    ax.legend(prop = {'size': 8})
    ax.grid()


def plotCNF(predicted_labels, true_labels, title = ' ') :
    confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['CN', 'AD'] )
    cm_display.plot()
    plt.title(title)
    plt.grid(False)
    plt.show()
