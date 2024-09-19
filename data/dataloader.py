import os
import librosa
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch

import sys
sys.path.append("..")
from config import *
from utils.global_constants import *



class SpeechDataset(Dataset):
    def __init__ (self, data, turn = 0):
        self.data = data
        self.tsr = 16e3 
        self.turn = turn


    def __getitem__ (self, item):
        path = self.data.path.values[item]
        label = self.data.label.values[item]
        file_name = os.path.basename(path)

        if self.turn == 0:
            inputs = self.file_to_array(path, self.tsr)
        elif self.turn == 1:
            inputs = self.data.embedding.values[item]

        return inputs, label, file_name


    def file_to_array (self, path, sampling_rate):
        array,_ = librosa.load(path, sr = sampling_rate)
        return array.copy()


    def __len__ (self):
        return len(self.data)
    


def collate_fn_padd(batch, feature_extractor, tsr):
    inputs, labels, file_names = zip(*batch)
    labels = torch.tensor(labels, dtype = torch.int64)
    padded_inputs = feature_extractor(inputs, padding = 'longest', sampling_rate = tsr,
                                        return_tensors = "pt").input_values
                                   
    return padded_inputs, labels, file_names



def get_dataloaders(dataset, fe):

    g = torch.Generator()
    g.manual_seed(seed)
    train_dl, valid_dl, test_dl = {}, {}, {}

    for turn, data_type in enumerate([SEG, SUB]):

        collate_fn = (lambda batch: collate_fn_padd(batch, feature_extractor = fe, tsr = 16e3)) if turn == 0 else None
 
        train_dl[data_type] = DataLoader(SpeechDataset(dataset[TRN][data_type], turn),
                                          batch_size = bs[turn], collate_fn=collate_fn,
                                            shuffle=True, generator=g, pin_memory=True)
        
        valid_dl[data_type] = DataLoader(SpeechDataset(dataset[VAL][data_type], turn),
                                          batch_size=2, collate_fn=collate_fn,
                                          shuffle=False,pin_memory=True)
        
        test_dl[data_type] = DataLoader(SpeechDataset(dataset[TST][data_type], turn),
                                         batch_size=2, collate_fn=collate_fn,
                                         shuffle=False, pin_memory=True)

    return train_dl, valid_dl, test_dl


