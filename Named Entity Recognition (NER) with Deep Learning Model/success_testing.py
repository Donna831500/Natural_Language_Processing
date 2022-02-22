# -*- coding: utf-8 -*-
"""success_testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16xBaM1joNjsTfPEXZd1Zx2AD9xrWLpC7
"""

########################################## testing task 1 ####################################################

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import sys

#from google.colab import drive
#drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = '/content/drive/My Drive/Colab Notebooks/HWK4_testing/'
argv_list = sys.argv
path = argv_list[1]

train = pd.read_table(path+'train',header=None,sep=' |\n', names=['index','word','tag'], doublequote = False, keep_default_na=False)
dev = pd.read_table(path+'dev',header=None,sep=' |\n', names=['index','word','tag'], doublequote = False, keep_default_na=False)
train['word'] = train['word'].astype(str)
dev['word'] = dev['word'].astype(str)

# ------------------------------Load Data------------------------------
# X
unique_words_list = sorted(list(train['word'].unique()))
num_unique_words = len(unique_words_list)
values = range(0,num_unique_words)
word_index_dict = dict(zip(unique_words_list, values))

# lower_word_index_dict
lower_unique_words_list = (map(lambda x: x.lower(), word_index_dict))
lower_word_index_dict = dict(zip(lower_unique_words_list, values))

# add unknown words
word_index_dict['<unk>'] = num_unique_words
word_index_dict['<unk_digit>'] = num_unique_words+1
word_index_dict['<unk_alnum>'] = num_unique_words+2

# load word embedding
embedding_weight = torch.load(path+'embedding_weight.pt')
embedding = nn.Embedding.from_pretrained(embedding_weight)

def add_features(s):
  try:
    lookup_tensor = torch.tensor([word_index_dict[s]], dtype=torch.long)
    word_embed = embedding(lookup_tensor)
    temp = word_embed.tolist()
    feature_list = temp[0]
  except:
    #try:
    #  lookup_tensor = torch.tensor([lower_word_index_dict[s]], dtype=torch.long)
    #except:
    if s[0].isdigit():
      lookup_tensor = torch.tensor([word_index_dict['<unk_digit>']], dtype=torch.long)
    elif s.isalnum():
      lookup_tensor = torch.tensor([word_index_dict['<unk_alnum>']], dtype=torch.long)
    else:
      lookup_tensor = torch.tensor([word_index_dict['<unk>']], dtype=torch.long)
    word_embed = embedding(lookup_tensor)
    temp = word_embed.tolist()
    feature_list = temp[0]

    #feature_list = [1]*100
  return feature_list

train['all_features'] = train['word'].apply(add_features)
dev['all_features'] = dev['word'].apply(add_features)

def getXY(index_l,feature_l):
  all_sentences = []
  current_sentence = []
  for i in range(0,len(index_l)):
    current_index = index_l[i]
    current_feature = feature_l[i]
    if current_index==1:
      all_sentences.append(torch.tensor(current_sentence)) # torch.tensor(  np.array(
      current_sentence = [current_feature]
    else:
      current_sentence.append(current_feature)
  all_sentences.append(torch.tensor(current_sentence)) # torch.tensor(  np.array(
  all_sentences = all_sentences[1:]
  return all_sentences

# X
train_index_list = train["index"].tolist()
train_feature_list = train["all_features"].tolist()
dev_index_list = dev["index"].tolist()
dev_feature_list = dev["all_features"].tolist()
X_train = getXY(train_index_list,train_feature_list)
X_dev = getXY(dev_index_list,dev_feature_list)

# Y
Y_train_df = pd.get_dummies(train.tag, prefix='y')
Y_dev_df = pd.get_dummies(dev.tag, prefix='y')
#Y_train_df['y_paddding'] = 0
#Y_dev_df['y_paddding'] = 0
Y_train_df['target']= Y_train_df.values.tolist()
Y_dev_df['target']= Y_dev_df.values.tolist()
train_target_list = Y_train_df["target"].tolist()
dev_target_list = Y_dev_df["target"].tolist()
#train_target_list
Y_train = getXY(train_index_list,train_target_list)
Y_dev = getXY(dev_index_list,dev_target_list)

train_dataset = list(zip(X_train,Y_train))
dev_dataset = list(zip(X_dev,Y_dev))

def pad_collate(batch):
  result = []
  all_x_batch = []
  all_y_batch = []
  sentence_len_list = []
  for each_tuple in batch:
    (x,y) = each_tuple
    all_x_batch.append(x)
    all_y_batch.append(y)
    sentence_len_list.append(len(x))
  xx_pad = pad_sequence(all_x_batch, batch_first=True, padding_value=0)
  yy_pad = pad_sequence(all_y_batch, batch_first=True, padding_value=0)
  return (xx_pad,yy_pad,sentence_len_list)

BATCH_SIZE = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,collate_fn=pad_collate, shuffle=False)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate, shuffle=False)

# Hyperparameters
embedding_dim = 100
num_layers = 1
hidden_size = 256
num_classes = len(train['tag'].unique()) # =9
LEARNING_RATE = 0.003
linear_output_size = 128

# Create a bidirectional LSTM
class BRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout1 = nn.Dropout(p=0.33)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.dropout2 = nn.Dropout(p=0.33)
        self.fc1 = nn.Linear(hidden_size * 2, linear_output_size)
        self.ELU = nn.ELU() #alpha=1.0, inplace=False
        self.fc2 = nn.Linear(linear_output_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        x = self.dropout1(x)

        out, _ = self.lstm(x, (h0, c0))
        #out = self.fc(out[:, -1, :])

        x = self.dropout2(out[:, -1, :])
        x = self.fc1(out) #x
        x = self.ELU(x)
        x = self.fc2(x)

        return x

brnn_model = torch.load(path+'blstm1.pt')
brnn_model = brnn_model.to(device)

# predict dev data
num_labels = len(train['tag'].unique())

def predict(model, dataloader):
    temp = list(Y_dev_df)
    tag_list = []
    for each_name in temp:
      tag_list.append(each_name[2:])

    prediction_list = []
    with torch.no_grad():
      for X_batch,y,sll in dataloader:
        X_batch = X_batch.float()
        X_batch, y = X_batch.to(device), y.to(device)
        output = model(X_batch)
        temp = output.tolist()

        for i in range(0,len(temp)):
          current_s_output = temp[i]
          current_s_len = sll[i]
          for j in range(0,current_s_len):
            current_w_output = current_s_output[j]
            max_index = current_w_output.index(max(current_w_output[:num_labels]))
            prediction_list.append(tag_list[max_index])

    return prediction_list

all_prediction = predict(brnn_model, dev_loader)
dev['pred'] = all_prediction

# output to task 1
t1_output_df = dev.drop('all_features', 1)
t1_output_list = t1_output_df.values.tolist()
first_flag = True
with open(path+'dev1_with_gold.out', 'w') as f:
    for each_output in t1_output_list:
        idx = each_output[0]
        word = each_output[1]
        gold = each_output[2]
        pred = each_output[3]

        if idx==1:
            if first_flag:
                first_flag = False
            else:
                f.write('\n')
        try:
          f.write(str(idx)+' '+word+' '+gold+' '+pred)
          f.write('\n')
        except:
          print(counter)
          print(idx)
          print(word)

f.close()

# output to task 1
t1_output_df = dev.drop('all_features', 1)
t1_output_list = t1_output_df.values.tolist()
first_flag = True
with open(path+'dev1.out', 'w') as f:
    for each_output in t1_output_list:
        idx = each_output[0]
        word = each_output[1]
        gold = each_output[2]
        pred = each_output[3]

        if idx==1:
            if first_flag:
                first_flag = False
            else:
                f.write('\n')
        try:
          f.write(str(idx)+' '+word+' '+pred)
          f.write('\n')
        except:
          print(counter)
          print(idx)
          print(word)

f.close()

####### task 1 test

# load test data
test = pd.read_table(path+'test',header=None,sep=' |\n', names=['index','word'], doublequote = False, keep_default_na=False)
test['all_features'] = test['word'].apply(add_features)

# X
test_index_list = test["index"].tolist()
test_feature_list = test["all_features"].tolist()
X_test = getXY(test_index_list,test_feature_list)
test_dataset = list(zip(X_test))

#################### data loader ####################
def test_pad_collate(batch):
  result = []
  all_x_batch = []
  sentence_len_list = []
  for each_tuple in batch:
    (x,) = each_tuple
    all_x_batch.append(x)
    sentence_len_list.append(len(x))

  xx_pad = pad_sequence(all_x_batch, batch_first=True, padding_value=0)
  return (xx_pad,sentence_len_list)

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,collate_fn=test_pad_collate, shuffle=False)

# predict test data
num_labels = len(train['tag'].unique())

temp = list(Y_dev_df)
tag_list = []
for each_name in temp:
  tag_list.append(each_name[2:])

def test_predict(model, dataloader, tag_list):
    prediction_list = []
    with torch.no_grad():
      for X_batch,sll in dataloader:
        X_batch = X_batch.float()
        X_batch = X_batch.to(device)
        output = model(X_batch)
        temp = output.tolist()

        for i in range(0,len(temp)):
          current_s_output = temp[i]
          current_s_len = sll[i]
          for j in range(0,current_s_len):
            current_w_output = current_s_output[j]
            max_index = current_w_output.index(max(current_w_output[:num_labels]))
            prediction_list.append(tag_list[max_index])

    return prediction_list

test_all_prediction = test_predict(brnn_model, test_loader,tag_list)

test['pred'] = test_all_prediction

# output to greedy.out
# df = df.drop('column_name', 1)
t1_output_df = test.drop('all_features', 1)
t1_output_list = t1_output_df.values.tolist()
first_flag = True
with open(path+'test1.out', 'w') as f:
    for each_output in t1_output_list:
        idx = each_output[0]
        word = each_output[1]
        #gold = each_output[2]
        pred = each_output[2]

        if idx==1:
            if first_flag:
                first_flag = False
            else:
                f.write('\n')
        try:
          f.write(str(idx)+' '+word+' '+pred)
          f.write('\n')
        except:
          print(idx)
          print(word)

f.close()

############################################## task 2 start #######################################################

################################################# TASK 2 ############################################################

# ------------------------------Load GloVe------------------------------
GloVe = pd.read_table(path+'glove.6B.100d.txt',header=None,sep=' |\n', doublequote = False)
GloVe[0] = GloVe[0].astype(str)
GloVe_word_list= GloVe[0].tolist()
GloVe.drop(GloVe.columns[0], axis=1, inplace=True)
GloVe['target']= GloVe.values.tolist()
GloVe_target_list = GloVe['target'].tolist()
#[str(i) for i in lst]
GloVe_word_list = [str(i) for i in GloVe_word_list]
GloVe_word_index_dict = dict(zip(GloVe_word_list, GloVe_target_list))

GloVe_lower_word_list = (map(lambda x: x.lower(), GloVe_word_list))
GloVe_lowcase_word_index_dict = dict(zip(GloVe_lower_word_list, GloVe_target_list))

def add_features(s):
  try:
    feature_list = GloVe_word_index_dict[s]
  except:
    try:
      feature_list = GloVe_lowcase_word_index_dict[s.lower()]
    except:
      if s[0].isdigit():
        lookup_tensor = torch.tensor([word_index_dict['<unk_digit>']], dtype=torch.long)
      elif s.isalnum():
        lookup_tensor = torch.tensor([word_index_dict['<unk_alnum>']], dtype=torch.long)
      else:
        lookup_tensor = torch.tensor([word_index_dict['<unk>']], dtype=torch.long)
      word_embed = embedding(lookup_tensor)
      temp = word_embed.tolist()
      feature_list = temp[0]

    #feature_list = [1]*100
  return feature_list

train['all_features'] = train['word'].apply(add_features)
dev['all_features'] = dev['word'].apply(add_features)

# X
train_index_list = train["index"].tolist()
train_feature_list = train["all_features"].tolist()
dev_index_list = dev["index"].tolist()
dev_feature_list = dev["all_features"].tolist()
X_train = getXY(train_index_list,train_feature_list)
X_dev = getXY(dev_index_list,dev_feature_list)

# Y
Y_train_df = pd.get_dummies(train.tag, prefix='y')
Y_dev_df = pd.get_dummies(dev.tag, prefix='y')
#Y_train_df['y_paddding'] = 0
#Y_dev_df['y_paddding'] = 0
Y_train_df['target']= Y_train_df.values.tolist()
Y_dev_df['target']= Y_dev_df.values.tolist()
train_target_list = Y_train_df["target"].tolist()
dev_target_list = Y_dev_df["target"].tolist()
#train_target_list
Y_train = getXY(train_index_list,train_target_list)
Y_dev = getXY(dev_index_list,dev_target_list)

train_dataset = list(zip(X_train,Y_train))
dev_dataset = list(zip(X_dev,Y_dev))

#################### data loader ####################
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,collate_fn=pad_collate, shuffle=False)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate, shuffle=False)

brnn_model_2 = torch.load(path+'blstm2.pt')
brnn_model_2 = brnn_model_2.to(device)

# predict dev data
num_labels = len(train['tag'].unique())
all_prediction = predict(brnn_model_2, dev_loader)
dev['pred'] = all_prediction

# output to greedy.out
# df = df.drop('column_name', 1)
t2_output_df = dev.drop('all_features', 1)
t2_output_list = t2_output_df.values.tolist()
first_flag = True
with open(path+'dev2_with_gold.out', 'w') as f:
    for each_output in t2_output_list:
        idx = each_output[0]
        word = each_output[1]
        gold = each_output[2]
        pred = each_output[3]

        if idx==1:
            if first_flag:
                first_flag = False
            else:
                f.write('\n')
        try:
          f.write(str(idx)+' '+word+' '+gold+' '+pred)
          f.write('\n')
        except:
          print(idx)
          print(word)

f.close()

# output to greedy.out
# df = df.drop('column_name', 1)
t2_output_df = dev.drop('all_features', 1)
t2_output_list = t2_output_df.values.tolist()
first_flag = True
with open(path+'dev2.out', 'w') as f:
    for each_output in t2_output_list:
        idx = each_output[0]
        word = each_output[1]
        gold = each_output[2]
        pred = each_output[3]

        if idx==1:
            if first_flag:
                first_flag = False
            else:
                f.write('\n')
        try:
          f.write(str(idx)+' '+word+' '+pred)
          f.write('\n')
        except:
          print(idx)
          print(word)

f.close()

####### task 2 test

# load test data
test = pd.read_table(path+'test',header=None,sep=' |\n', names=['index','word'], doublequote = False, keep_default_na=False)
test['all_features'] = test['word'].apply(add_features)

# X
test_index_list = test["index"].tolist()
test_feature_list = test["all_features"].tolist()
X_test = getXY(test_index_list,test_feature_list)
test_dataset = list(zip(X_test))

#################### data loader ####################
def test_pad_collate(batch):
  result = []
  all_x_batch = []
  sentence_len_list = []
  for each_tuple in batch:
    (x,) = each_tuple
    all_x_batch.append(x)
    sentence_len_list.append(len(x))

  xx_pad = pad_sequence(all_x_batch, batch_first=True, padding_value=0)
  return (xx_pad,sentence_len_list)

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,collate_fn=test_pad_collate, shuffle=False)

# predict test data
num_labels = len(train['tag'].unique())

temp = list(Y_dev_df)
tag_list = []
for each_name in temp:
  tag_list.append(each_name[2:])

def test_predict(model, dataloader, tag_list):
    prediction_list = []
    with torch.no_grad():
      for X_batch,sll in dataloader:
        X_batch = X_batch.float()
        X_batch = X_batch.to(device)
        output = model(X_batch)
        temp = output.tolist()

        for i in range(0,len(temp)):
          current_s_output = temp[i]
          current_s_len = sll[i]
          for j in range(0,current_s_len):
            current_w_output = current_s_output[j]
            max_index = current_w_output.index(max(current_w_output[:num_labels]))
            prediction_list.append(tag_list[max_index])

    return prediction_list

test_all_prediction = test_predict(brnn_model_2, test_loader,tag_list)

test['pred'] = test_all_prediction

# output to greedy.out
# df = df.drop('column_name', 1)
t2_output_df = test.drop('all_features', 1)
t2_output_list = t2_output_df.values.tolist()
first_flag = True
with open(path+'test2.out', 'w') as f:
    for each_output in t2_output_list:
        idx = each_output[0]
        word = each_output[1]
        #gold = each_output[2]
        pred = each_output[2]

        if idx==1:
            if first_flag:
                first_flag = False
            else:
                f.write('\n')
        try:
          f.write(str(idx)+' '+word+' '+pred)
          f.write('\n')
        except:
          print(idx)
          print(word)

f.close()
