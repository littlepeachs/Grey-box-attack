import argparse
import time
import math
import os
import torch
import torch.nn as nn
from torch import optim
# import data 
import pandas as pd
import model
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from ppl import calculate_ppl
import bert_score

parser = argparse.ArgumentParser(description="pytorch emotion classify")
parser.add_argument("--data",type=str,default="c:\\Users\\86135\\Desktop\\h2_CPU\\L2_NLP_pipeline_pytorch\\data",help="get data")
parser.add_argument('--model',type=str,default='LSTM',help='catagory of model:RNN_RELU,RNN_TANH,LSTM,GRU')
parser.add_argument('--emsize',type=int,default=200,help='size of word embedding')
parser.add_argument('--nhid',type=int,default=200,help='hidden units number per layer')
parser.add_argument('--nlayers',type=int,default=2,help='layer number')
parser.add_argument('--lr',type=float,default=1e-5,help='init_learing rate') ##so big?
parser.add_argument('--clip',type=float,default=0.25,help='gradient clip')
parser.add_argument('--epochs',type=int,default=6,help='upper epoch limit')
parser.add_argument('--batch_size',type=int,default=64,help='batch_size')
parser.add_argument('--dropout',type=float,default=0.2,help='dropout apply to layer')
parser.add_argument('--cuda',action="store_true",help='use cuda')
parser.add_argument('--save0',type=str,default='model_pretrain_0.pt',help='save model pretrain0')
parser.add_argument('--save1',type=str,default='model_pretrain_1.pt',help='save model pretrain1')

parser.add_argument('--pretrain',type=int,default=1,help='pretrain model') #pretrain=0:train on multinli_1.0 pretrain=1:train on manmade-data


def tokenize(ori_dataset,gen_dataset):
    encoded_ori_dataset = tokenizer(ori_dataset,padding="max_length",truncation=True, return_tensors="pt",max_length=512)
    encoded_gen_dataset = tokenizer(gen_dataset,padding="max_length",truncation=True, return_tensors="pt",max_length=512)
    return encoded_ori_dataset,encoded_gen_dataset

def get_batch(ori_datas,gen_datas,i,bsz):
    max_len = ori_datas.size(0)
    if(max_len<=i+bsz):
        ori_data = ori_datas[i:]
        gen_data = gen_datas[i:] #max_len-bsz:max_len
        return ori_data.to(device) ,gen_data.to(device)
    else:
        ori_data = ori_datas[i:i+bsz]
        gen_data = gen_datas[i:i+bsz]
        return ori_data.to(device) ,gen_data.to(device)

def train():
    bsz = 8 ##
    model.train()
    total_loss=0.0
    total_acc = 0.0
    step=0
    ppl=0
    total_ppl=0
    for i in range(0,train_ori_id.size(0)-1,bsz): #use val_data to get fast test
        ori_data,gen_data = get_batch(train_ori_id,train_gen_id,i,bsz)
        en_input = model.embed(ori_data)
        de_input = model.embed(gen_data)
        de_output=gen_data
        model.zero_grad()
        output,train_loss = model(en_input,de_input,de_output)
        # print("success")
        # loss = criterion(output,label)
        if args.pretrain==1:
            output = torch.argmax(output,2)
            
            for k in range(len(output)):

                ori_sentence = tokenizer.decode(ori_data[k])
                gen_sentence = tokenizer.decode(output[k])
                ori_sentence = process_sentence(ori_sentence)
                gen_sentence = process_sentence(gen_sentence)
                if k==0:
                    print("ori_sentence:{}".format(ori_sentence))
                    print("gen_sentence:{}".format(gen_sentence))
                ppl= calculate_ppl(LM_model,LM_tokenizer,gen_sentence)
                if k%5==0:
                    print("k:{} \n ppl : {}".format(k,ppl))
                total_ppl+=ppl
            total_ppl/=len(output)
        total_loss=total_ppl+train_loss

        total_loss.backward(retain_graph=True)  #retain_graph=True
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()
        # total_loss+=loss.item()
    #     step+=1
    #     if((i/bsz)%100==0):
    #         print("batch_num{}, batch_loss:{}".format(i,loss.item()))
    # total_loss/=step
    # print("train_loss: {:.6f}".format(total_loss))
    # print("total_acc: {:.6f}".format(total_acc))

def evaluate(valid_ori_id,valid_gen_id,flag):  #flag==1 mean test
    eval_batch_size =32
    model.eval()
    total_loss=0.0
    total_acc=0.0
    step = 0
    # ntokens=len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0,valid_ori_id.size(0)-1,eval_batch_size):
            ori_data,gen_data = get_batch(valid_ori_id,valid_gen_id,i,eval_batch_size)
            en_input = model.embed(ori_data)
            de_input = model.embed(gen_data)
            de_output=gen_data
            model.zero_grad()
            output,loss = model(en_input,de_input,de_output)
            total_loss+=loss
            step+=1
    total_loss/=step
    if(flag==0):
        print("val_loss: {}".format(total_loss))
    if(flag==1):
        print("\n")
        print("test_loss: {}".format(total_loss))
    return total_loss

def process_sentence(sentence):
    ls = sentence.split()
    posi =0
    for i in range(len(ls[0])):
        if ls[0][posi]!='>':
            posi+=1
        else:
            break
    ls[0] = ls[0][posi+1:]
    posi =0
    for i in range(len(ls[-1])):
        if ls[-1][posi]!='<':
            posi+=1
        else:
            break
    ls[-1] = ls[-1][:posi]
    str=' '
    return str.join(ls)

def decode():
    eval_batch_size =32
    model.eval()
    total_loss=0.0
    total_acc=0.0
    step = 0
    # ntokens=len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0,100,eval_batch_size):
            ori_data,gen_data = get_batch(valid_ori_id,valid_gen_id,i,eval_batch_size)
            en_input = model.embed(ori_data)
            de_input = model.embed(gen_data)
            de_output=gen_data
            model.zero_grad()
            output,loss = model(en_input,de_input,de_output)
            output = torch.argmax(output,2)[0]
            ori_sentence = tokenizer.decode(ori_data[0])
            gen_sentence = tokenizer.decode(output)
            ori_sentence = process_sentence(ori_sentence)
            gen_sentence = process_sentence(gen_sentence)
            print("ori_sentence:{}".format(ori_sentence))
            print("gen_sentence:{}".format(gen_sentence))

args = parser.parse_args()

torch.manual_seed(1111)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING:you have cuda device")
# device = torch.device("cuda"if args.cuda else"cpu")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)



if args.pretrain==0:
    dataset = pd.read_json("multinli_1.0/multinli_1.0_train.jsonl",lines=True)
    dataset = dataset.loc[dataset["gold_label"]=="entailment"]
    ori_train= dataset["sentence1"].to_list()
    gen_train= dataset["sentence2"].to_list()

    dataset = pd.read_json("multinli_1.0/multinli_1.0_dev_matched.jsonl",lines=True)
    dataset = dataset.loc[dataset["gold_label"]=="entailment"]
    ori_dev= dataset["sentence1"].to_list()
    gen_dev= dataset["sentence2"].to_list()

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_ori_dataset,train_gen_dataset = tokenize(ori_train,gen_train)
    valid_ori_dataset,valid_gen_dataset = tokenize(ori_dev,gen_dev)

    train_ori_id = train_ori_dataset["input_ids"]
    train_gen_id = train_gen_dataset["input_ids"]
    valid_ori_id=valid_ori_dataset["input_ids"]
    valid_gen_id=valid_gen_dataset["input_ids"]

    in_features = 1024
    ninp = 1024
    hidden_size = 1024
    ntoken = tokenizer.vocab_size

    encoder = model.Encoder(in_features,hidden_size).to(device)
    decoder = model.Decoder(in_features,hidden_size).to(device)
    model = model.Seq2seq(encoder, decoder, in_features, hidden_size,ntoken,ninp).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        print("epoch: {}".format(epoch+1))
        train()
        evaluate(valid_ori_id,valid_gen_id,0)
        decode()
    with open(args.save0,'wb') as f:
        torch.save(model,f)

elif args.pretrain==1:
    with open(args.save0,'rb') as f:
        model = torch.load(f)
    dataset = pd.read_excel("train.xlsx")
    dataset = dataset.loc[dataset["gold_label"]==2]
    ori_train= dataset["sentence1"].to_list()
    gen_train= dataset["sentence2"].to_list()

    dataset = pd.read_excel("dev.xlsx")
    dataset = dataset.loc[dataset["gold_label"]==2]
    ori_dev= dataset["sentence1"].to_list()
    gen_dev= dataset["sentence2"].to_list()

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_ori_dataset,train_gen_dataset = tokenize(ori_train,gen_train)
    valid_ori_dataset,valid_gen_dataset = tokenize(ori_dev,gen_dev)

    LM_model = AutoModelForMaskedLM.from_pretrained('roberta-base')
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    LM_tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    train_ori_id = train_ori_dataset["input_ids"]
    train_gen_id = train_gen_dataset["input_ids"]
    valid_ori_id=valid_ori_dataset["input_ids"]
    valid_gen_id=valid_gen_dataset["input_ids"]
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        print("epoch: {}".format(epoch+1))
        train()
        evaluate(valid_ori_id,valid_gen_id,0)
        decode()

    # with open(args.save1,'wb') as f:
    #     torch.save(model,f)


