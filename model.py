import math
import torch
import torch.nn as nn
import torch.nn.functional as f
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=in_features, hidden_size=hidden_size, dropout=0.5, num_layers=1)  # encoder

    def forward(self, enc_input):
        seq_len, batch_size, embedding_size = enc_input.size()
        h_0 = torch.rand(1, batch_size, self.hidden_size).to(device)
        c_0 = torch.rand(1, batch_size, self.hidden_size).to(device)
        # en_ht:[num_layers * num_directions,Batch_size,hidden_size]
        encode_output, (encode_ht, decode_ht) = self.encoder(enc_input, (h_0, c_0))
        return encode_output, (encode_ht, decode_ht)


class Decoder(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.crition = nn.CrossEntropyLoss()
        self.fc = nn.Linear(hidden_size, in_features)
        self.decoder = nn.LSTM(input_size=in_features, hidden_size=hidden_size, dropout=0.5, num_layers=1)  # encoder

    def forward(self, enc_output, dec_input):
        (h0, c0) = enc_output
        # en_ht:[num_layers * num_directions,Batch_size,hidden_size]
        de_output, (_, _) = self.decoder(dec_input, (h0, c0))
        return de_output

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, in_features, hidden_size,ntoken,ninp):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = nn.Embedding(ntoken,ninp)  # need a dropout
        self.in_features = in_features 
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, ntoken)
        self.crition = nn.CrossEntropyLoss()
    
    def embed(self,input):
        enc = self.embedding(input)
        return enc

    def forward(self, enc_input, dec_input, dec_output):
        enc_input = enc_input.permute(1, 0, 2)  # [seq_len,Batch_size,embedding_size]
        dec_input = dec_input.permute(1, 0, 2)  # [seq_len,Batch_size,embedding_size]
        # output:[seq_len,Batch_size,hidden_size]
        _, (ht, ct) = self.encoder(enc_input)  # en_ht:[num_layers * num_directions,Batch_size,hidden_size]
        de_output = self.decoder((ht, ct), dec_input)  # de_output:[seq_len,Batch_size,in_features]
        output = self.fc(de_output)
        output = output.permute(1, 0, 2)
        loss = 0
        for i in range(len(output)):  # 对seq的每一个输出进行二分类损失计算
            loss += self.crition(output[i], dec_output[i])
        loss/=len(output)
        return output, loss