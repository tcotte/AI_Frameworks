from imblearn.over_sampling import RandomOverSampler
from keras_preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertAdam, BertForSequenceClassification
import pandas as pd


def balance_dataset(train_df, train_label):
    split_t_dataset = train_df.merge(train_label, how='left', on='Id')
    for i in range(28):
        is_cat = split_t_dataset["Category"] == i
        if len(split_t_dataset[is_cat]) > 50000:
            split_t_dataset = split_t_dataset.drop(split_t_dataset[is_cat][50000:].index, axis=0)
        if len(split_t_dataset[is_cat]) < 4000:
            split_t_dataset = pd.concat([split_t_dataset, split_t_dataset[is_cat], split_t_dataset[is_cat]],
                                        ignore_index=True)
            split_t_dataset = split_t_dataset.reset_index(drop=True)

    split_t_dataset = split_t_dataset.reset_index(drop=True)

    train_df = split_t_dataset[["description", "Id"]]
    train_labels = split_t_dataset[["Category", "Id"]]
    return[train_df, train_labels]


def tokenize_plus_attention(df, MAX_LEN=150):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in df]

    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    return [input_ids, attention_masks]
