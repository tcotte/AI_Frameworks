import time
import numpy as np
import pandas as pd
import tensorflow as tf
import io
import tensorflow as tf
import torch
from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import io
from sklearn.model_selection import train_test_split
import cleantext as ct
from processing import balance_dataset, tokenize_plus_attention
from training import train, flat_accuracy, predict
import argparse

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## Argument
DATA_DIR = ""
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--data_dir', type=str,
                    default=DATA_DIR + "Data")
parser.add_argument('--results_dir', type=str,
                    default=DATA_DIR + "results")
parser.add_argument('--model_dir', type=str,
                    default=DATA_DIR + "model/")
args = parser.parse_args()

# Load data
DATA_PATH = args.data_dir
train_df = pd.read_json(DATA_PATH + "/train.json")
test_df = pd.read_json(DATA_PATH + "/test.json")
train_label = pd.read_csv(DATA_PATH + "/train_label.csv")
categories = pd.read_csv(DATA_PATH + "/categories_string.csv")

# Data processing
train_df, train_labels = balance_dataset(train_df, train_label)

preprocessor = Pipeline(steps=[
    ('minuscule', ct.Minuscule("description")),
    ('url', ct.RemoveURL("description")),
    ('html', ct.RemoveHTML("description")),
    ('rm_ponctutation', ct.RemovePonctuationBert("description")),  # we keep the "." and the figures
    ('tokens', ct.Tokens(feature="description", stemming=False)),
    ('lit', ct.ListIntoSentence("description")),
    ('bert', ct.SentenceBert("description")),  # we add CLS and SEP
    ('drop_columns', ct.DropColumns(["Id"]))
])

train_df = preprocessor.fit_transform(train_df)

# Train validation split
train_df, validation_df, train_labels, validation_labels = train_test_split(train_df, train_labels.Category,
                                                                            random_state=2018, test_size=0.1)
# Bert tokenization
train_inputs, train_masks = tokenize_plus_attention(train_df.description)
validation_inputs, validation_masks = tokenize_plus_attention(validation_df.description)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels.values)
validation_labels = torch.tensor(validation_labels.values)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Data Generator
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.batch_size)

# Model definition
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)

# Learning
print("Start learning")
ts = time.time()
# Store our loss and accuracy for plotting
train_losses = []
for epoch in range(args.epochs):
    train_losses.append(train(model, train_dataloader, device, optimizer, epoch))

te = time.time()
t_learning = te - ts

print("Start predicting")
ts = time.time()
v_logits, v_labels, eval_acc = predict(model, validation_dataloader, device)
te = time.time()
t_predicting = te - ts


# Save the fine-tuning model
torch.save(model.state_dict(), args.model_dir + 'bert_base_balanced')

# Save the results
print("Save results")
d = {'learning_time': t_learning, 'prediction_time': t_predicting, 'loss_train': train_losses[-1], 'accuracy_test' : eval_acc}
results_df = pd.DataFrame(data=d, index=[0])
results_file = args.results_dir+'/results.csv'
with open(results_file, mode='w') as f:
    results_df.to_csv(f)
