import os.path
import os 
import numpy as np
from collections import defaultdict 
from sklearn.metrics import f1_score 
from transformers import Wav2Vec2FeatureExtractor
import torch.nn.functional as F
from torch import nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import transformers
import logging


from data import get_dataloaders
from models import get_model
from config import *
from utils import *
 


class Trainer():

    def __init__ (self, dataset):

        set_seed()
        self.reports = Reports()
        self.dataset = dataset

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\n-> {self.device} is available \n')

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(transformer_checkpoint)
        print('-> FeatureExtractor loaded successfully \n')

        self.train_dataloader, self.valid_dataloader, self.test_dataloader = get_dataloaders(dataset, self.feature_extractor)
        print(f'-> Number of train batches: {len(self.train_dataloader[SEG])} (batch size = {bs[0]})')
        print(f'   Number of validaion batches: {len(self.valid_dataloader[SEG])} (batch size = 2) \n'); 


    def update_lr(self, optimizer, scheduler, loss, epoch):
        pre_lr = optimizer.param_groups[0]['lr']
        scheduler.step(loss)
        pos_lr = optimizer.param_groups[0]['lr']
        if pre_lr != pos_lr:
            print(f'Learning rate reduced at epoch {epoch + 1}!')


    def train (self, model, dataloader, optimizer, epoch):
        model.train()
        criterion = nn.CrossEntropyLoss()

        pred_labels, true_labels, losses = [], [], []
        for step, batch in enumerate(dataloader):

            optimizer.zero_grad(set_to_none=True)

            (inputs, labels) = [t.to(self.device) for t in batch[:2]]
            outputs = model(inputs)
            output_probs = F.softmax(outputs, dim = 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pred_labels.extend(torch.argmax(output_probs, dim = 1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            if step % 10 == 0:
                print(f'\r[{epoch + 1}][{step}] -> Loss = {np.mean(losses):.4f}', end='')

        return np.mean(losses), pred_labels, true_labels



    def evaluate (self, model, dataloader):
        model.eval()
        criterion = nn.CrossEntropyLoss()

        pred_labels, true_labels, pred_probs, losses, ids = [], [], [], [], []
        segment_probs = defaultdict(lambda: [])
        for batch in dataloader:

            # deactivate autograd
            with torch.no_grad():

                (inputs, labels) = [t.to(self.device) for t in batch[:2]]
                file_names = batch[2]
                outputs = model(inputs)
                output_probs = F.softmax(outputs, dim = 1)

            # store normalized prediction score for each segment
            for i, name in enumerate(file_names):
                segment_probs[name.split('.')[0]].append(output_probs[i].detach().cpu().numpy())

            losses.append(criterion(outputs, labels).item())
            pred_probs.extend(output_probs.detach().cpu().tolist())
            pred_labels.extend(torch.argmax(output_probs, dim = 1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            ids.extend(file_names)

        return np.mean(losses), pred_probs, pred_labels, true_labels, ids, segment_probs



    def train_and_evaluate(self):
        self.segment_loss_list, self.segment_f1_list, self.voting_f1_list, self.embed_f1_list = [], [], [], []

        segment_pred_probs = self.train_on_segments()

        self.voting(segment_pred_probs)

        self.embed_based()

        self.reports.final_results[SEG]['best_epoch'] = np.argmax(np.array(self.segment_f1_list)[:,1])+ 1
        self.reports.final_results[VOTING]['best_epoch'] = np.argmax(self.voting_f1_list)+ 1
        self.reports.final_results[EMBED_BASED]['best_epoch'] = np.argmax(self.embed_f1_list) + 1

        plot_training(np.array(self.segment_loss_list), np.array(self.segment_f1_list), ['Train' , 'Validation'],
            'T:{} - M: {} - S: {}'.format(0, transformer_checkpoint.split('/')[1], seed))



    def train_on_segments(self):
        
        model = get_model(0, self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr[0], weight_decay = 5e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, min_lr=5e-6, threshold=0.05)
        
        segment_pred_probs = []
        print('\n\n### **Training model on Segments**')
        for epoch in range(epochs[0]):

            t_loss, t_pred_labels, t_true_labels = self.train( model,
                                                               self.train_dataloader[SEG],
                                                               optimizer,
                                                               epoch)

            v_loss, _, v_pred_labels, v_true_labels, _, segment_probs = self.evaluate( model,
                                                                                    self.valid_dataloader[SEG])

            segment_pred_probs.append(segment_probs)

            t_f1 = f1_score(t_true_labels, t_pred_labels)
            v_f1 = f1_score(v_true_labels, v_pred_labels)

            self.segment_loss_list.append([t_loss, v_loss])
            self.segment_f1_list.append([t_f1, v_f1])

            print(f'\r[{epoch+1}]: (Train) L: {t_loss:.4f}, F: {t_f1*100:.2f}' +
                  f' >< (VAl) L: {v_loss:.4f}, F: {v_f1*100:.2f}')

            self.update_lr(optimizer, scheduler, t_loss, epoch)
            if (epoch+1) % 3 == 0: scheduler.threshold =  0.1 * t_loss

            # save the one epoch trained model
            torch.save(model.state_dict(), f'{epoch}.pt')

        return segment_pred_probs


            
    def voting(self, segment_pred_probs):
        for segment_probs in segment_pred_probs:
            _, predicted_labels, true_labels, _ = self.merge_probs(segment_probs, self.dataset[VAL][SUB])
            self.voting_f1_list.append(f1_score(true_labels, predicted_labels))



    def embed_based(self):
        print('\n### **Embed_Based: Training model on subjects**')
        best_embed_f1 = -1
        for super_epoch in range(epochs[0]):

            print('*Extracting Embeddings*', end='')
            for set, dataloader in list(zip([TRN, VAL], [self.train_dataloader, self.valid_dataloader])):

                self.dataset[set][SUB]['embedding'] = self.dataset[set][SUB]['key'].map( extract_embeddings(
                    dataloader[SEG],
                    get_model(0, self.device, f'{super_epoch}.pt', True),
                    self.device))

            self.dataset[TST][SUB] = self.dataset[VAL][SUB].copy()

            # Create new dataloaders
            self.new_train_dataloader, self.new_valid_dataloader, self.new_test_dataloader = get_dataloaders(self.dataset,
                                                                                                             self.feature_extractor)

            # Initiate model and related modules
            model = get_model(1, self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr[1], weight_decay = 5e-3)

            loss_list, f1_list = [], []
            for epoch in range(epochs[1]):

                t_loss, t_pred_labels, t_true_labels = self.train(model,
                                                                  self.new_train_dataloader[SUB],
                                                                  optimizer,
                                                                  epoch)

                # Evaluate model on validation data
                v_loss, _, v_pred_labels, v_true_labels, segment_ids, _ = self.evaluate(model,
                                                                                     self.new_valid_dataloader[SUB])

                t_f1 = f1_score(t_true_labels, t_pred_labels)
                v_f1 = f1_score(v_true_labels, v_pred_labels)

                loss_list.append([t_loss, v_loss])
                f1_list.append([t_f1, v_f1])

                if v_f1 > best_embed_f1:
                    best_embed_f1 = v_f1
                    torch.save(model.state_dict(), best_embed_model_path)
                    self.reports.final_results[EMBED_BASED]['best_epoch'] = super_epoch

            self.embed_f1_list.append(np.max(np.array(f1_list)[:,1]))
            # Reporting the results
            print('\r'+ 
                  f'[{super_epoch+1}]: Segment-f1: {self.segment_f1_list[super_epoch][1]*100:.2f} '+
                  f'Voting-f1: {self.voting_f1_list[super_epoch]*100:.2f} '+
                  f'Embed_f1: {self.embed_f1_list[super_epoch]*100:.2f}')



    def merge_probs(self, segment_probs, dataset_df):

        subjects_probs = {}
        for file_name, probs in segment_probs.items():
            subjects_probs[file_name] = np.mean(np.array(probs), axis = 0)

        predicted_probs, predicted_labels, true_labels, ids = [], [], [], []
        for index, row in dataset_df.iterrows():
            file_name = os.path.basename(row["path"]).split('.')[0]
            label = row["label"]
            if file_name in subjects_probs:
                predicted_probs.append(subjects_probs[file_name])  # AD labels encoded to 1
                predicted_labels.append(np.argmax(subjects_probs[file_name]))
                true_labels.append(label)
                ids.append(file_name)

        return predicted_probs, predicted_labels, true_labels, ids


    def testing(self, mode, path, saveandplot= False):

        if mode == SEG:
            segment_model = get_model(0, self.device, path, True)

            _, pred_probs, pred_labels, true_labels, ids, segment_probs = self.evaluate(segment_model,
                                                                                        self.test_dataloader[SEG])

            if saveandplot:
                save_classification_reports(self.reports, mode, pred_probs, pred_labels, true_labels)
                add_eval_probs(self.reports,mode, pred_probs, pred_labels, true_labels, ids)
                plotCNF(pred_labels, true_labels, 'Confusion Matrix - Segment - Test data')

            return segment_probs

        if mode == VOTING:

            segment_probs = self.testing(SEG, path)
            pred_probs, pred_labels, true_labels, ids = self.merge_probs(segment_probs, self.dataset[TST][SUB])

            if saveandplot:
                save_classification_reports(self.reports, mode, pred_probs, pred_labels, true_labels)
                add_eval_probs(self.reports,mode, pred_probs, pred_labels, true_labels, ids)
                # plotCNF(pred_labels, true_labels, 'Confusion Matrix - ScoreBased - Test data')

        if mode == EMBED_BASED:

            segment_model = get_model(0, self.device, str(self.reports.final_results[EMBED_BASED]['best_epoch'])+'.pt', True)
            sub_model  = get_model(1, self.device, path, True)

            self.dataset[TST][SUB]['embedding'] = self.dataset[TST][SUB]['key'].map(extract_embeddings(
                self.test_dataloader[SEG],
                segment_model,
                self.device))

            _, _, self.new_test_dataloader = get_dataloaders(self.dataset,self.feature_extractor)
            _, pred_probs, pred_labels, true_labels, ids, _ = self.evaluate(sub_model,self.new_test_dataloader[SUB])

            if saveandplot:
                save_classification_reports(self.reports, mode, pred_probs, pred_labels, true_labels)
                add_eval_probs(self.reports,mode, pred_probs, pred_labels, true_labels, ids)
                # plotCNF(pred_labels, true_labels, 'Confusion Matrix - EmbedBased - Test data')

            return self.reports.final_results, self.reports.evaluation_results, self.reports.evaluation_probs