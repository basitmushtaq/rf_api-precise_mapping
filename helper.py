from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
import numpy as np
import json


def expand_embeddings(embeddings_dict, num_binned_features=39):
    """
    Expand embeddings into separate DataFrame columns.
    """

    list_embeddings_dict = [embeddings_dict] * num_binned_features
    print(len(list_embeddings_dict))
    embeddings_df = pd.DataFrame(
        list_embeddings_dict, columns=["Description", "embeddings"]
    )
    embed_df = pd.DataFrame(
        embeddings_df["embeddings"].tolist(), index=embeddings_df.index
    )
    print("length embed df", embed_df)
    embed_df.columns = [f"embedding_{i}" for i in range(embed_df.shape[1])]

    return pd.concat([embeddings_df.drop(columns=["embeddings"]), embed_df], axis=1)


def generate_bert_embeddings(text):
    # Load pre-trained model tokenizer (vocabulary) and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Encode the text, adding the special tokens needed for BERT
    encoded_input = tokenizer(text, return_tensors="pt")

    # Get the output from the model
    with torch.no_grad():
        output = model(**encoded_input)

    # The last_hidden_state is a sequence of hidden states of the last layer of the model
    # Obtaining the mean of all token embeddings to represent the sentence embedding
    embeddings = output.last_hidden_state.mean(dim=1).squeeze()
    return {"text": text, "embeddings": embeddings.tolist()}


def prepare_inference_input(text, training_columns):
    """
    Prepares a DataFrame for model inferencing based on the text description and feature bins.
    Also reorders columns to match the training data structure.
    Returns:
    pd.DataFrame: A DataFrame ready for model inferencing, with columns reordered.
    """
    with open("default_column_values.json", "r") as f:
        default = json.load(f)
    # print(default)
    test_dict_list = []
    for i in default.keys():
        if i == "index":
            continue
        temp = default.copy()

        temp[i] = True
        temp["Description"] = text
        test_dict_list.append(temp)
    embeddings_dict = generate_bert_embeddings(text)
    embeddings_df = expand_embeddings(embeddings_dict)
    input_df = pd.DataFrame(test_dict_list)
    input_df = pd.concat([input_df, embeddings_df], axis=1)
    with open("training_columns.json", "r") as f:
        training_columns = json.load(f)
    input_df = input_df[training_columns]

    return input_df
