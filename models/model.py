import torch
from torch import nn
from utils import set_seed
from transformers import AutoModel
import torch.nn.functional as F
from config import *
import transformers
import logging

transformers.logging.set_verbosity_error()



class SegmentNetwork(nn.Module):

    def __init__ (self):
        super(SegmentNetwork, self).__init__()
        set_seed()
        self.transformer_model = AutoModel.from_pretrained(transformer_checkpoint)
        embed_size = self.transformer_model.config.hidden_size
        self.projector = nn.Linear(embed_size, segnet_midsize)
        self.classifier = nn.Linear(segnet_midsize, num_labels)


    def freeze_feature_encoder (self):
        self.transformer_model.feature_extractor._freeze_parameters()


    def freeze_all(self):
        self.freeze_feature_encoder()
        for param in self.transformer_model.parameters():
            param.requires_grad = False


    def freeze_except (self, except_layer):
        self.freeze_feature_encoder()
        for i, model_child in enumerate(self.transformer_model.children()):
            if i != 2:
                for param in model_child.parameters():
                    param.requires_grad = False
            else:
                for j, module in enumerate(model_child.children()):
                    if j != 3:
                        for param in module.parameters():
                            param.requires_grad = False
                    else:
                        for num_layer, child in enumerate(module.children()):
                            if (num_layer != (except_layer - 1)):
                                for param in child.parameters():
                                    param.requires_grad = False


    def average_pooling(self, hidden_states):
        return torch.mean(hidden_states, dim = 1)


    def forward (self, input_waveform):
        output_embeddings = self.transformer_model(input_waveform).last_hidden_state
        x = self.average_pooling(output_embeddings)
        x = self.projector(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x



class SubjectNetwork(nn.Module):

    def __init__ (self):
        super(SubjectNetwork, self).__init__()
        set_seed()
        self.drop = nn.Dropout(p = 0.15)
        self.projector = nn.Linear(subnet_insize, subnet_midsize)
        self.classifier = nn.Linear(subnet_midsize, num_labels)


    def forward (self, x):
        x = F.relu(self.projector(x))
        x = self.drop(x)
        x = self.classifier(x)
        return x
    


def get_model(turn, device, weights_path= None, freeze_all=False):
    if turn == 0:
        model = SegmentNetwork().to(device)
        model.freeze_feature_encoder()
        if freeze_all: model.freeze_all()

    elif turn == 1:
        model = SubjectNetwork().to(device)

    if weights_path!= None:
            model.load_state_dict(torch.load(weights_path, weights_only=True))

    return model