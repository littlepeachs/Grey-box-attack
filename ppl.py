import pandas as pd
import numpy as np
import bert_score
from bert_score import score,get_idf_dict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

# dataset = pd.read_excel("annotate_data/processed_data.xlsx")
# dataset.head()

# ori_sent = dataset["ori_sent"].tolist()
# attack_sent = dataset["attack_sent"].tolist()

# ori_sample = ori_sent[0]
# attack_sample = attack_sent[0]


# Load pre-trained model (weights)
def calculate_ppl(LM_model,LM_tokenizer,sample):
    sentence = sample
    tokenize_input = LM_tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([LM_tokenizer.convert_tokens_to_ids(tokenize_input)])
    sen_len = len(tokenize_input)
    sentence_loss = 0.

    for i, word in enumerate(tokenize_input):
        # add mask to i-th character of the sentence
        tokenize_input[i] = '[MASK]' #it is normal to have so big ppl
        mask_input = torch.tensor([LM_tokenizer.convert_tokens_to_ids(tokenize_input)])

        output = LM_model(mask_input)

        prediction_scores = output[0]
        softmax = nn.Softmax(dim=0)
        ps = softmax(prediction_scores[0, i]).log()
        word_loss = ps[tensor_input[0, i]]
        sentence_loss += word_loss

        tokenize_input[i] = word
    ppl = -sentence_loss/sen_len #torch.exp(-sentence_loss/sen_len)
    return ppl
