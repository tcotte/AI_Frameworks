import torch
import argparse
import cleantext as ct
import pandas as pd
from processing import tokenize_plus_attention
from pytorch_pretrained_bert import BertForSequenceClassification
from training import predict
import numpy as np

# Device
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Argument
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=16)

DATA_DIR = ""
parser.add_argument('--data_dir', type=str,
                    default=DATA_DIR + "data")
parser.add_argument('--results_dir', type=str,
                    default=DATA_DIR + "results")
parser.add_argument('--model_dir', type=str,
                    default=DATA_DIR + "model")

args = parser.parse_args()

# Load data
DATA_PATH = args.data_dir
df = pd.read_json(DATA_PATH + "/test.json")

# data processing
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

test_df = preprocessor.fit_transform(df[:10])

# import model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
args_str = "bert_epochs_%d_batch_size_%d" %(args.epochs, args.batch_size)
model.load_state_dict(torch.load(args.model_dir +"/"+ args_str))


# Bert tokenization
test_inputs, test_masks = tokenize_plus_attention(test_df.description)

t_prediction_inputs = torch.tensor(test_inputs)
t_prediction_masks = torch.tensor(test_masks)

# we split the test set for memory allocation
test_data = TensorDataset(t_prediction_inputs, t_prediction_masks)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

# Predictions
total_test_logits = []
# Evaluate data for one epoch
for batch in test_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    t_input_ids, t_input_mask = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        test_logits = model(torch.tensor(t_input_ids).to(device).long(), token_type_ids=None, attention_mask=t_input_mask)

        # Move logits and labels to CPU
        test_logits = test_logits.detach().cpu().numpy()

        # store logits and labels
        total_test_logits.extend(test_logits)

predictions = np.argmax(total_test_logits, axis=1)

# Put predictions in csv file
df["Category"] = predictions
response_file = df[["Id", "Category"]]
response_file.to_csv(args.results_dir+"/prediction_"+args_str+".csv", index=False)
