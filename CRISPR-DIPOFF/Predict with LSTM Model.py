#!/usr/bin/env python
# coding: utf-8

# # Directory and Device Setup

# In[ ]:


import os
import random
import torch
import numpy as np
import copy
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# In[ ]:


# Required if your the experiment on Colba and use google drive
# from google.colab import drive
# drive.mount('/gdrive')


# In[ ]:


# Replace according to your environment
root_path = '/home/jsinohui/CRISPR-DIPOFF/'
model_dir = root_path + "Trained Models/"
input_dir = root_path + "Sample Input/"
output_dir = root_path + "Sample Output/"

input_filename = "CRISPR-DIPOFF_training_data_circleseq_full_v2.csv"
output_filename = "crispr-dipoff_out.csv"
CHANNEL_SIZE = 4


# In[ ]:


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# device = 'cpu'
print(device)


# # Helper Functions

# ## Data Encoding

# In[ ]:


def encoder(RNAseq, order=['A','T','C','G']):
    lookup_table = {order[0]:[1,0,0,0],
                    order[1]:[0,1,0,0],
                    order[2]:[0,0,1,0],
                    order[3]:[0,0,0,1]}
    encoded = np.zeros((len(RNAseq),len(order)))

    for i in range(len(RNAseq)):
        nu = RNAseq[i]
        if nu in lookup_table:
            encoded[i] = np.array(lookup_table[nu])
        else:
            print("Exception: Unindentified Nucleotide")

    return encoded

def decoder(encoded, order=['A','T','C','G']):
    RNAseq = ''

    for i in range(encoded.shape[0]):
        idx = np.where(encoded[i]==1)[0][0] #first occurance only
        RNAseq += order[idx]

    return RNAseq

def superpose(encoded1, encoded2):
    if(len(encoded1) != len(encoded2)):
        print("Size Mismatch")
        return encoded1

    superposed = np.zeros(encoded1.shape)

    for i in range(len(encoded1)):
        for j in range(len(encoded1[i])):
            if encoded1[i][j] == encoded2[i][j]:
                superposed[i][j] = encoded1[i][j]
            else:
                superposed[i][j] = encoded1[i][j] + encoded2[i][j]
    return superposed

def superposeWithDirection(encoded1, encoded2):
    if(len(encoded1) != len(encoded2)):
        print("Size Mismatch")
        return encoded1

    superposed = np.zeros((encoded1.shape[0],encoded1.shape[1]+1))

    for i in range(len(encoded1)):
        for j in range(len(encoded1[i])):
            if encoded1[i][j] == encoded2[i][j]:
                superposed[i][j] = encoded1[i][j]
            else:
                superposed[i][j] = encoded1[i][j] + encoded2[i][j]
                superposed[i][-1] = encoded1[i][j]
    return superposed

def testEncDec():
    sgRNA = 'ACTGGG'
    print("Original: ", sgRNA)
    print("Encoded:")
    encoded = encoder(sgRNA)
    print(encoded)
    decoded = decoder(encoded)
    print("Decoded: ",decoded)


def testSuperpose():
    sgRNA = "ACTGGG"
    DNA = "GCTGGC"
    print('sgRNA: ', sgRNA)
    print('DNA  : ', DNA)

    encoded1 = encoder(sgRNA)
    encoded2 = encoder(DNA)

    superposed = superpose(encoded1, encoded2)
    print(superposed)

def testSuperposeWithDirection():
    sgRNA = "GACTGGGC"
    DNA = "AGCTGGCG"
    print('sgRNA: ', sgRNA)
    print('DNA  : ', DNA)

    encoded1 = encoder(sgRNA)
    encoded2 = encoder(DNA)

    superposed = superposeWithDirection(encoded1, encoded2)
    print(superposed)

testEncDec()
print()
testSuperpose()
print()
testSuperposeWithDirection()


# In[ ]:


def get_encoded_data(df, channel_size = 4):
    enc_targets = []
    enc_off_targets = []
    enc_superposed = []
    enc_superposed_with_dir = []
    labels = []

    for i in range(df.shape[0]):
        df_row = df.iloc[i]
        target = encoder(df_row['sgRNA'])
        off_target = encoder(df_row['targetDNA'])
        superposed = superpose(target, off_target)
        superposed_with_dir = superposeWithDirection(target, off_target)

        enc_targets.append(target)
        enc_off_targets.append(off_target)
        enc_superposed.append(superposed)
        enc_superposed_with_dir.append(superposed_with_dir)
        labels.append(df_row['label'])

        if i%1000 == 0:
            print(i+1,"/",df.shape[0],"done")

    print(len(enc_targets))
    print(len(enc_off_targets))
    print(len(enc_superposed))
    print(len(superposed_with_dir))
    print(len(labels))

    if channel_size == 4:
        return enc_superposed, labels
    else:
        return enc_superposed_with_dir. labels


# In[ ]:


def load_data(filename, channel_size = 4):
    data = pd.read_csv(input_dir + filename)
    print("Data Loaded")
    print(data.shape)
    print(data.head(5))

    print("Adding a dummy label column with all 0s")
    data["label"] = 0
    print(data["label"].value_counts())
    print("Encoding Sequences...")
    data_x, data_y = get_encoded_data(data, channel_size)
    print("Encoding Complete.")
    return data, data_x, data_y


# ## Model Definition

# In[ ]:


class RNN_Model_Generic(nn.Module):
    def __init__(self, config, model_type):
        super(RNN_Model_Generic,self).__init__()
        # emb_size=256, hidden_size=128, hidden_layers=3, output=2

        self.model_type = model_type
        self.vocab_size = config["vocab_size"]
        self.emb_size = config["emb_size"]
        self.hidden_size = config["hidden_size"]
        self.lstm_layers = config["lstm_layers"]
        self.bi_lstm = config["bi_lstm"]
        self.reshape = config["reshape"]

        self.number_hidden_layers = config["number_hidder_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.hidden_layers = []

        self.hidden_shape = self.hidden_size*2 if self.bi_lstm else self.hidden_size

        self.embedding = None
        if self.vocab_size > 0:
            self.embedding = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)


        if model_type == "LSTM":
            self.lstm = nn.LSTM(self.emb_size, self.hidden_size, num_layers=self.lstm_layers,
                            batch_first=True, bidirectional=self.bi_lstm)
        elif model_type == "GRU":
            self.lstm= nn.GRU(self.emb_size, self.hidden_size, num_layers=self.lstm_layers,
                           batch_first=True, bidirectional=self.bi_lstm)
        else:
            self.lstm= nn.RNN(self.emb_size, self.hidden_size, num_layers=self.lstm_layers,
                           batch_first=True, bidirectional=self.bi_lstm)

        start_size = self.hidden_shape

        self.relu = nn.ReLU
        # self.dropout = nn.Dropout(self.dropout_prob)

        for i in range(self.number_hidden_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(start_size, start_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_prob)))

            start_size = start_size // 2

        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output = nn.Linear(start_size,2)


    def forward(self,x):
        # added for captum's prediction
        softmax = nn.Softmax(dim=1)

        dir = 2 if self.bi_lstm else 1
        h = torch.zeros((self.lstm_layers*dir, x.size(0), self.hidden_size)).to(device)
        c = torch.zeros((self.lstm_layers*dir, x.size(0), self.hidden_size)).to(device)

        if self.embedding is not None:
            x = x.type(torch.LongTensor).to(device)
            x = self.embedding(x)
        elif self.reshape:
            x = x.view(x.shape[0],x.shape[1],1)

        if self.model_type == "LSTM":
            x, (hidden, cell) = self.lstm(x, (h,c))
        else:
            x, hidden = self.lstm(x, h)

        x = x[:, -1, :]

        # print(x.shape)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            # print(x.shape)
        x = self.output(x)

        #This line has been added only for model evaluation. Should be removed for training
        x  = softmax(x)
        # print(x.shape)
        return x


# In[ ]:


def load_best_rnn_model():
    model_weights = model_dir + "best_lstm_model.pth"
    model_config = {
        'vocab_size': 0,
        'emb_size': 4,
        'hidden_size': 512,
        'lstm_layers': 1,
        'bi_lstm': True,
        'number_hidder_layers': 2,
        'dropout_prob': 0.4,
        'reshape': False,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.00010
    }
    model = RNN_Model_Generic(model_config, "LSTM").to(device)

    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_weights))

    model.eval()
    return model


# ## Tester Functions

# In[ ]:


class TrainerDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs= inputs
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.Tensor(self.inputs[idx]), self.targets[idx]


# In[ ]:


def tester(model, test_x, test_y):
    test_dataset = TrainerDataset(test_x, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model.eval()
    results = []

    with torch.no_grad():
        for test_features, test_labels in test_dataloader:
            outputs = model(test_features.to(device)).detach().to("cpu")
            results.extend(outputs)

    pred_y = np.array([y[1].item() for y in results])
    pred_y_list = []

    for x in pred_y:
        if(x>0.5):
            pred_y_list.append(1)
        else:
            pred_y_list.append(0)

    return pred_y_list, pred_y


# # Make Predictions LSTM Model

# In[ ]:


# loading and encoding data
df, data_x, data_y = load_data(input_filename, CHANNEL_SIZE)
data_x = np.array(data_x)
data_y = np.array(data_y)

print(data_x.shape)
print(data_y.shape)


# In[ ]:


#loading the model
model = load_best_rnn_model()
print(model)


# In[ ]:


# get model output
# data_y is dummy with all 0s
predictions, probabilities = tester(model, data_x, data_y)


# In[ ]:


df = df.drop(["label"],axis=1)
df["predictions"] = predictions
df["probabilities"] = probabilities
print(df.shape)
print(df.head(5))
print(df["predictions"].value_counts())


# In[ ]:


# Save output
df.to_csv(output_dir + output_filename, index = False)

