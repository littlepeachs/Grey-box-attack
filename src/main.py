import argparse
import time
import math
import os
import torch
import torch.nn as nn
from torch import optim
import numpy as np
# import data 
import pandas as pd
import model
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM,AutoModelForSequenceClassification,T5ForConditionalGeneration
import torch.nn.functional as F
from tqdm import tqdm
from pandas.core.frame import DataFrame

parser = argparse.ArgumentParser(description="pytorch emotion classify")
parser.add_argument("--data",type=str,default="c:\\Users\\86135\\Desktop\\h2_CPU\\L2_NLP_pipeline_pytorch\\data",help="get data")
parser.add_argument('--model',type=str,default='LSTM',help='catagory of model:RNN_RELU,RNN_TANH,LSTM,GRU')
parser.add_argument('--emsize',type=int,default=200,help='size of word embedding')
parser.add_argument('--nhid',type=int,default=200,help='hidden units number per layer')
parser.add_argument('--nlayers',type=int,default=2,help='layer number')
parser.add_argument('--lr',type=float,default=2e-6,help='init_learing rate') ##so big?
parser.add_argument('--clip',type=float,default=0.25,help='gradient clip')
parser.add_argument('--epochs',type=int,default=5,help='upper epoch limit')
parser.add_argument('--batch_size',type=int,default=8,help='batch_size')
parser.add_argument('--dropout',type=float,default=0.2,help='dropout apply to layer')
parser.add_argument('--cuda',type = bool,default=True,help='use cuda')
parser.add_argument('--save0',type=str,default='model_pretrain_0.pt',help='save model pretrain0')
parser.add_argument('--save1',type=str,default='model_pretrain_1.pt',help='save model pretrain1')
parser.add_argument('--attack',type=int,default=1,help='save model pretrain1') # default attack on roberta-large sst2

parser.add_argument('--pretrain',type=int,default=-1,help='pretrain model') #pretrain=0:train on multinli_1.0 pretrain=1:train on manmade-data


def tokenize(ori_dataset,gen_dataset=None):
    if gen_dataset==None:
        encoded_ori_dataset = tokenizer(ori_dataset,padding="longest",truncation=True, return_tensors="pt",max_length=128)
        return encoded_ori_dataset
        encoded_ori_dataset = tokenizer(ori_dataset,padding="max_length",truncation=True, return_tensors="pt",max_length=128)
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

def soft_crossentropy(logit,target,num_label,label_smooth):
    logprobs = F.log_softmax(logit, dim=1)	# softmax + log
    target = F.one_hot(target,num_label)	# 转换成one-hot
    target = (1.0-label_smooth)*target + label_smooth/num_label
    loss = -1*torch.sum(target*logprobs, 1)
    return loss.mean()

def train():
    bsz = 8 ##
    model.train()
    total_loss=0.0
    step=0
    loss_func = nn.CrossEntropyLoss()
    for i in range(0,train_ori_id.size(0)-1,bsz): #use val_data to get fast test
        ori_data,gen_data = get_batch(train_ori_id,train_gen_id,i,bsz)
        en_input = model.embed(ori_data)
        de_input = model.embed(gen_data)
        de_output=gen_data
        model.zero_grad()
        output,_,train_loss,sim_loss = model(en_input,de_input,de_output)
        # print("success")
        # loss = loss_func(output,gen_data)
        if step%20==0:
            if args.pretrain==1:
                output = torch.argmax(output,2)
                
                for k in range(len(output)):

                    ori_sentence = tokenizer.decode(ori_data[k])
                    gen_sentence = tokenizer.decode(output[k])
                    ori_sentence = process_sentence(ori_sentence)
                    gen_sentence = process_sentence(gen_sentence)
                    if k%5==0:
                        print("ori_sentence:{}".format(ori_sentence))
                        print("gen_sentence:{}".format(gen_sentence))
        loss=train_loss*sim_loss # +total_ppl

        loss.backward()  #retain_graph=True
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()
        total_loss+=loss.item()
        step+=1
        if step%1000==0:
            print("ave_loss: {}".format(total_loss/step))
    total_loss/=step
    print("train_loss: {:.6f}".format(total_loss))


def adv_train(text,attention_mask,label):
    # classify_model = AutoModelForSequenceClassification.from_pretrained("howey/roberta-large-sst2").to(device)
    # fluency_model = AutoModelForSequenceClassification.from_pretrained("cointegrated/roberta-large-cola-krishna2020").to(device)
    # classify_embed_mat = classify_model.roberta.embeddings.word_embeddings.weight # because the output of the attack model is float-tensor, must separate the embedding layer
    # fluency_embed_mat = fluency_model.roberta.embeddings.word_embeddings.weight
    # classify_embed_layer = classify_model.roberta.embeddings
    # classify_encoder = classify_model.roberta.encoder
    # classify_classifier = classify_model.classifier
    # fluency_embed_layer = fluency_model.roberta.embeddings
    # fluency_encoder = fluency_model.roberta.encoder
    # fluency_classifier = fluency_model.classifier
    with open("t5_cls_model.pt",'rb') as f:
        classify_model = torch.load(f).to(device)
    classify_embed_mat = classify_model.shared.weight # because the output of the attack model is float-tensor, must separate the embedding layer
    classify_embed_layer = classify_model.shared
    classify_encoder = classify_model.encoder
    classify_encoder = classify_model.decoder
    classsiy_lmhead = classify_model.lm_head
    model.train()
    total_loss=0.0
    step=0
    bsz=4
    loss_func = nn.CrossEntropyLoss()
    # awl = AutomaticWeightedLoss(3)
    # model.to(device)
    optimizer = optim.Adam([
                {'params': model.parameters(),"lr":args.lr,'weight_decay': 1e-7},
                # {'params': awl.parameters(), 'weight_decay': 0}	
            ])
    verbalizer = ["terrible","great"]
    verbalizer_ids = [tokenizer.encode(verbalizer[0])[0],tokenizer.encode(verbalizer[1])[0]]
    for i in tqdm(range(0,len(text),bsz)): #use val_data to get fast test
        batch_text,batch_label,batch_attention_mask = get_batch(text,label,i,bsz,attention_mask)
        # en_input = model.embed(batch_text)
        # de_input = model.embed(batch_text)
        # de_output=gen_data
        model.zero_grad()
        ori_embed = model.shared(batch_text)
        ori_sen_embed = torch.mean(ori_embed,dim=1)
        output = model(batch_text,attention_mask =batch_attention_mask,labels=batch_text,decoder_attention_mask = batch_attention_mask)
        ppl_loss = output.loss
        output = F.gumbel_softmax(output.logits,hard=True,tau=0.01)
        decode_output = torch.argmax(output,dim=2)
        decode_attention_masks = []
        for sen_num in range(decode_output.size(0)):
            decode_attention_mask =torch.where(decode_output[sen_num]==1,decode_output[sen_num],0)
            decode_attention_mask = decode_attention_mask^1
            decode_attention_masks.append(decode_attention_mask.float().unsqueeze(0))
        decode_attention_masks =torch.cat(decode_attention_masks).to(device)


        victim_embed_input = torch.matmul(output,classify_embed_mat)
        victim_sen_embed = torch.mean(victim_embed_input,dim=1)

        cos_similarity = torch.cosine_similarity(ori_sen_embed,victim_sen_embed,dim=1)
        sim_loss = torch.exp(-torch.mean(cos_similarity))

        # victim_embed_output = classify_embed_layer(inputs_embeds=victim_embed_input)
        target_text = ["<pad> <extra_id_0> %s" % verbalizer[batch_label[j]] for j in range(len(batch_label))]
        decoded_inputs = tokenizer(target_text, pad_to_max_length=True, return_tensors="pt",max_length=8).to(device)

        adv_logits = classify_model(inputs_embeds = victim_embed_input,attention_mask = decode_attention_masks,decoder_input_ids = decoded_inputs['input_ids'],decoder_attention_mask = decoded_inputs['attention_mask']).logits[:,1,:]
        # adv_loss = loss_func(output_logits,batch_label)
        verbalizer_logits = adv_logits[:,verbalizer_ids]
        adv_loss = soft_crossentropy(verbalizer_logits,batch_label,2,0.05)

        # FL_embed_input = torch.matmul(output,fluency_embed_mat)
        # FL_embed_output = fluency_embed_layer(inputs_embeds=FL_embed_input)
        # FL_hidden_state = fluency_encoder(FL_embed_output)
        # FL_logits = fluency_classifier(FL_hidden_state.last_hidden_state)
        # FL_score = torch.mean(torch.matmul(F.softmax(FL_logits,dim=1),torch.tensor([0,1],dtype = torch.float32).to(device)))
        # FL_score =FL_score**(1/3)
        # loss=0.1*adv_loss+sim_loss+FL_score+2*adv_loss*sim_loss*FL_score
        loss = adv_loss+2*sim_loss+0.4*ppl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()
        total_loss += loss.item()
        step+=1
        # if step%100==0:
        #     compare_attack_effect(batch_text,batch_attention_mask,batch_label,classify_model,output,adv_logits)

    total_loss/=step
    print("train_loss: {:.6f}".format(total_loss))

def compare_attack_effect(batch_text,attention_mask,batch_label,victim_model,output,adv_logits):
    # loss_func = nn.CrossEntropyLoss()
    # vocab_size = 50625
    verbalizer = ["terrible","great"]
    verbalizer_ids = [tokenizer.encode(verbalizer[0])[0],tokenizer.encode(verbalizer[1])[0]]
    target_text = ["<pad> <extra_id_0> %s" % verbalizer[batch_label[j]] for j in range(len(batch_label))]
    decoded_inputs = tokenizer(target_text, pad_to_max_length=True, return_tensors="pt",max_length=8).to(device)

    ori_output= victim_model(batch_text,attention_mask,decoded_inputs['input_ids'],decoded_inputs['attention_mask']).logits[:,1,:]
    verbalizer_logits = adv_logits[:,verbalizer_ids]
    ori_label = torch.argmax(verbalizer_logits,dim=1)
    ori_acc = 0
    for i in range(len(batch_label)):
        ori_acc+=(ori_label[i]==batch_label[i]^1).item()
    ori_acc/=len(batch_label)

    ori_sentence_ls = []
    for i in range(len(ori_label)):
        ori_sentence_ls.append(process_sentence(tokenizer.decode(batch_text[i])))
    print("#################### Before the Grey-Box Attack, The original sentence ############################")
    print(ori_sentence_ls[0:5])
    # print("\n")
    print("#################### Before the Grey-Box Attack, The model acc is #################################")
    print(ori_acc)
    print("###################################################################################################")

    adv_label = torch.argmax(adv_logits,dim=1)
    attack_acc = 0
    for i in range(len(batch_label)):
        attack_acc+=(adv_label[i]==batch_label[i]^1).item()
    attack_acc/=len(batch_label)
    
    
    attack_sentence_ls = []
    output = torch.argmax(output,dim=2)
    for i in range(len(batch_label)):
        for j in range(output.size(1)):
            if output[i][j]==1:
                posi=j
                break
        attack_sentence_ls.append(tokenizer.decode(output[i][:posi]))
    print("#################### After the Grey-Box Attack, The attack sentence #################################\n")
    print(attack_sentence_ls[0:5])
    print("#################### After the Grey-Box Attack, The model acc is #################################")
    print(attack_acc)
    print("###################################################################################################")

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
            output,_,train_loss,sim_loss = model(en_input,de_input,de_output)
            total_loss+=train_loss.item()*sim_loss.item()
            step+=1
    total_loss/=step
    if(flag==0):
        print("val_loss: {}".format(total_loss))
    if(flag==1):
        print("\n")
        print("test_loss: {}".format(total_loss))
    return total_loss

def adv_evaluate(text,attention_mask,label,epoch):
    # classify_model = AutoModelForSequenceClassification.from_pretrained("howey/roberta-large-sst2").to(device)
    with open("t5_cls_model.pt",'rb') as f:
        classify_model = torch.load(f).to(device)

    verbalizer = ["terrible","great"]
    verbalizer_ids = [tokenizer.encode(verbalizer[0])[0],tokenizer.encode(verbalizer[1])[0]]
    
    model.eval()
    total_loss=0.0
    bsz=1
    loss_func = nn.CrossEntropyLoss()
    ori_acc = 0
    total_dev_acc = 0.0
    softmax_func = nn.Softmax(dim=1)
    adv_sentences = []
    step=0
    sen_id = []
    with torch.no_grad():
        for i in range(len(text)):
            target_text = ["<pad> <extra_id_0> %s" % verbalizer[label[i]]]
            decoded_inputs = tokenizer(target_text, pad_to_max_length=True, return_tensors="pt",max_length=8).to(device)
            input_ids = tokenizer(text[i], return_tensors="pt").to(device)
            beam_output = model.generate(input_ids['input_ids'].to(device),max_length=60, num_beams=15,early_stopping=True)[0]
            adv_sen = tokenizer.decode(beam_output, skip_special_tokens=True)
            attack_example=tokenizer(adv_sen,pad_to_max_length=True, return_tensors="pt",max_length=80).to(device)
            adv_logits = classify_model(attack_example['input_ids'],attack_example['attention_mask'],decoded_inputs['input_ids'],decoded_inputs['attention_mask']).logits[:,1,:]
            adv_verbalizer_logits = adv_logits[:,verbalizer_ids]
            adv_probability = softmax_func(adv_verbalizer_logits)
            if step%50==0:
                ori_logits = classify_model(input_ids['input_ids'],input_ids['attention_mask'],decoded_inputs['input_ids'],decoded_inputs['attention_mask']).logits[:,1,:]
                ori_verbalizer_logits = ori_logits[:,verbalizer_ids]
                ori_probability = softmax_func(ori_verbalizer_logits)
                print("########## Evaluation for {}th example ###########".format(i),flush=True)
                print("ori_sentence:{}".format(text[i]),flush=True)
                print("adv_sentence:{}".format(adv_sen),flush=True)
                print("groud truth label : {}".format(label[i]),flush=True)
                print("ori_probability:{}".format(ori_probability),flush=True)
                print("adv_probability:{}".format(adv_probability),flush=True)
            step+=1
            predict = torch.argmax(adv_verbalizer_logits,dim=1)
            if predict[0]==label[i]:
                total_dev_acc+=1
            else:
                adv_sentences.append(adv_sen)
                sen_id.append(i)
    # total_dev_acc/=len(text)
    Dict = {'sen_id':sen_id,'adv_sentence':adv_sentences}
    data = DataFrame(Dict)
    data.to_csv("success_sample_{}.tsv".format(epoch),sep='\t')
    total_dev_acc/=len(text)
    print("################## Evaluation ###############")
    # print("Before adv training, the ori_acc goes down to: {}".format(ori_acc))
    print("After adv training, the total_dev_acc goes down to: {}".format(total_dev_acc))
             

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
            output,_,train_loss,sim_loss = model(en_input,de_input,de_output)
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
        print("##################### training mode ############################")
        train()
        print("##################### evaluate mode ############################")
        evaluate(valid_ori_id,valid_gen_id,0)
        print("##################### decode the sentences ############################")
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

    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    train_ori_dataset,train_gen_dataset = tokenize(ori_train,gen_train)
    valid_ori_dataset,valid_gen_dataset = tokenize(ori_dev,gen_dev)

    # LM_model = AutoModelForMaskedLM.from_pretrained('roberta-base')
    
    # Load pre-trained model tokenizer (vocabulary)
    # LM_tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    train_ori_id = train_ori_dataset["input_ids"]
    train_ori_mask = train_ori_dataset["attention_mask"]
    train_gen_id = train_gen_dataset["input_ids"]
    train_gen_mask = train_gen_dataset["attention_mask"]
    valid_ori_id=valid_ori_dataset["input_ids"]
    valid_ori_mask = valid_ori_dataset["attention_mask"]
    valid_gen_id=valid_gen_dataset["input_ids"]
    valid_gen_mask = valid_gen_dataset["attention_mask"]
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        print("epoch: {}".format(epoch+1))
        train()
        evaluate(valid_ori_id,valid_gen_id,0)
        decode()

    with open(args.save1,'wb') as f:
        torch.save(model,f)

if args.attack==1:

    # with open("t5_cls_model.pt",'rb') as f:
    #     model = torch.load(f)
    # model.to(device)
    model = T5ForConditionalGeneration.from_pretrained('t5-large',cache_dir ='/home/liwentao/seq2seq/t5_model/t5-large').to(device)
    dataset = pd.read_csv("SST-2/train.tsv",sep='\t')
    dev_dataset = pd.read_csv("SST-2/dev.tsv",sep='\t')
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    ntoken = tokenizer.vocab_size
    text = dataset["sentence"].to_list()
    label= dataset["label"].to_list()
    dev_text = dev_dataset['sentence'].to_list()
    dev_label = dev_dataset["label"].to_list()

    for i in range(len(label)):  # attack target label
        label[i] = label[i]^1
    # for i in range(len(dev_label)):
    #     dev_label[i] = dev_label[i]^1
    input = tokenize(text)
    input_id = input["input_ids"]
    attention_mask = input["attention_mask"]
    label = torch.tensor(label)

    dev_input = tokenize(dev_text)
    dev_input_id = dev_input["input_ids"]
    dev_attention_mask = dev_input["attention_mask"]
    dev_label = torch.tensor(dev_label)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("######## hyperparameter:")
    print("batch_size:{}".format(args.batch_size))
    print("epochs:{}".format(args.epochs))
    print("learning_rate:{}".format(args.lr))
    print("loss_formulation: AutoWeight(3loss)")
    for epoch in range(args.epochs):
        adv_train(input_id,attention_mask,label)
        adv_evaluate(dev_text,dev_attention_mask,dev_label,epoch)
        # decode()
