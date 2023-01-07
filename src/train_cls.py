import pandas as pd
from transformers import AutoTokenizer,T5ForConditionalGeneration
import torch
import torch.nn as nn
import argparse
from torch import optim

parser = argparse.ArgumentParser(description="pytorch emotion classify")
parser.add_argument('--cuda',type = bool,default=True,help='use cuda')
parser.add_argument('--lr',type=int,default=5e-5,help='save model pretrain0')
parser.add_argument('--batch_size',type=int,default=16,help='save model pretrain1')
parser.add_argument('--epoch',type=int,default=5,help='save model pretrain1') # default attack on roberta-large sst2

parser.add_argument('--pretrain',type=int,default=-1,help='pretrain model')
args = parser.parse_args()

device = torch.device("cuda:7")

def tokenize(tokenizer,ori_dataset,gen_dataset=None):
    if gen_dataset==None:
        encoded_ori_dataset = tokenizer(ori_dataset,padding="max_length",truncation=True, return_tensors="pt",max_length=128)
        return encoded_ori_dataset
    encoded_ori_dataset = tokenizer(ori_dataset,padding="max_length",truncation=True, return_tensors="pt",max_length=128)
    encoded_gen_dataset = tokenizer(gen_dataset,padding="max_length",truncation=True, return_tensors="pt",max_length=128)
    return encoded_ori_dataset,encoded_gen_dataset

def get_batch(ori_datas,gen_datas,i,bsz,ori_masks=None,gen_masks=None):
    max_len = ori_datas.size(0)
    if ori_masks==None:
        if(max_len<=i+bsz):
            ori_data = ori_datas[i:]
            gen_data = gen_datas[i:]
            return ori_data.to(device) ,gen_data.to(device)
        else:
            ori_data = ori_datas[i:i+bsz]
            gen_data = gen_datas[i:i+bsz]
            return ori_data.to(device) ,gen_data.to(device)
    else:
        if(max_len<=i+bsz):
            ori_data = ori_datas[i:]
            gen_data = gen_datas[i:] #max_len-bsz:max_len
            ori_mask = ori_masks[i:]
            return ori_data.to(device) ,gen_data.to(device),ori_mask.to(device)
        else:
            ori_data = ori_datas[i:i+bsz]
            gen_data = gen_datas[i:i+bsz]
            ori_mask = ori_masks[i:i+bsz]
            return ori_data.to(device) ,gen_data.to(device),ori_mask.to(device)

def _train(input_id,attention_mask,label):
    model.train()
    total_loss=0.0
    step=0
    loss_func = nn.CrossEntropyLoss()
    verbalizer = ["terrible","great"]
    verbalizer_ids = [tokenizer.encode(verbalizer[0])[0],tokenizer.encode(verbalizer[1])[0]]
    for i in range(0,label.size(0)-1,args.batch_size): #use val_data to get fast test
        enc_input,batch_label,batch_attention_mask =get_batch(input_id,label,i,args.batch_size,attention_mask)
        target_text = ["<pad> <extra_id_0> %s" % verbalizer[batch_label[j]] for j in range(len(batch_label))]
        decoded_inputs = tokenizer(target_text, pad_to_max_length=True, return_tensors="pt",max_length=8).to(device)
        model.zero_grad()
        token_logits = model(enc_input,batch_attention_mask,decoded_inputs['input_ids'],decoded_inputs['attention_mask']).logits[:,1,:]
        verbalizer_logits = token_logits[:,verbalizer_ids]
        loss = loss_func(verbalizer_logits,batch_label)
        loss.backward()  #retain_graph=True
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.25)
        optimizer.step()
        total_loss+=loss.item()
        step+=1
        if step%1000==0:
            print("ave_loss: {}".format(total_loss/step))
    total_loss/=step
    print("train_loss: {:.6f}".format(total_loss))

def _evaluate(input_id,attention_mask,label):
    model.eval()
    total_acc=0.0
    step=0
    verbalizer = ["terrible","great"]
    verbalizer_ids = [tokenizer.encode(verbalizer[0])[0],tokenizer.encode(verbalizer[1])[0]]
    with torch.no_grad():
        for i in range(0,label.size(0)-1,args.batch_size): #use val_data to get fast test
            enc_input,batch_label,batch_attention_mask =get_batch(input_id,label,i,args.batch_size,attention_mask)
            target_text = ["<pad> <extra_id_0> %s" % verbalizer[batch_label[j]] for j in range(len(batch_label))]
            decoded_inputs = tokenizer(target_text, pad_to_max_length=True, return_tensors="pt",max_length=8).to(device)
            model.zero_grad()
            token_logits = model(enc_input,batch_attention_mask,decoded_inputs['input_ids'],decoded_inputs['attention_mask']).logits[:,1,:]
            verbalizer_logits = token_logits[:,verbalizer_ids]
            predict = torch.argmax(verbalizer_logits,dim=1)
            for j in range(len(predict)):
                if predict[j]==batch_label[j]:
                    total_acc+=1
            step+=len(predict)
    total_acc/=step
    print("total_acc: {:.6f}".format(total_acc))
    return total_acc

tokenizer = AutoTokenizer.from_pretrained("t5-large")
    
model = T5ForConditionalGeneration.from_pretrained('t5-large',cache_dir ='/home/liwentao/seq2seq/t5_model/t5-large').to(device)
optimizer = optim.Adam([
        {'params': model.parameters(),"lr":args.lr,'weight_decay': 1e-7}
    ])

def train_cls():
    dataset = pd.read_csv("SST-2/train.tsv",sep='\t')
    dev_dataset = pd.read_csv("SST-2/dev.tsv",sep='\t')
    ntoken = tokenizer.vocab_size
    text = dataset["sentence"].to_list()
    label= dataset["label"].to_list()
    dev_text = dev_dataset['sentence'].to_list()
    dev_label = dev_dataset["label"].to_list()

    # for i in range(len(label)):  # attack target label
    #     label[i] = label[i]^1
    # for i in range(len(dev_label)):
    #     dev_label[i] = dev_label[i]^1
    input = tokenize(tokenizer,text)
    input_id = input["input_ids"]
    attention_mask = input["attention_mask"]
    label = torch.tensor(label)

    dev_input = tokenize(tokenizer,dev_text)
    dev_input_id = dev_input["input_ids"]
    dev_attention_mask = dev_input["attention_mask"]
    dev_label = torch.tensor(dev_label)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    
    print("######## hyperparameter:")
    print("batch_size:{}".format(args.batch_size))
    print("epochs:{}".format(args.epoch))
    print("learning_rate:{}".format(args.lr))
    input_id = input["input_ids"]
    attention_mask = input["attention_mask"]
    best_acc = 0.0
    for epoch in range(args.epoch):
        _train(input_id,attention_mask,label)
        total_acc = _evaluate(dev_input_id,dev_attention_mask,dev_label)
        if total_acc>best_acc:
            best_acc = total_acc
            with open("t5_cls_model.pt",'wb') as f:
                torch.save(model,f)

train_cls()