
import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.init as init
from config import *

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, model_name='rnn'):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.model_name = model_name
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        if model_name == 'rnn':
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
        elif model_name == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif model_name == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        else:
            raise ValueError('Unknown model_name {}' %(model_name))
        
        
    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        if self.model_name in ['rnn', 'gru']:
            hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
            if USE_CUDA: 
                hidden = hidden.cuda()
            return hidden
        elif self.model_name == 'lstm':
            h0 = torch.zeros(self.n_layers, 1, self.hidden_size)
            c0 = torch.zeros(self.n_layers, 1, self.hidden_size)
            if USE_CUDA: 
                h0 = h0.cuda()
                c0 = c0.cuda()
            return h0, c0


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, model_name='rnn'):
        super(DecoderRNN, self).__init__()
        
        # Keep parameters for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.model_name = model_name
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        if model_name == 'rnn':
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        elif model_name == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        elif model_name == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout_p)

        else:
            raise ValueError('Unknown model_name {}' %(model_name))
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, word_input, last_hidden):
        # Note: we run this one step at a time        
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        rnn_output, hidden = self.rnn(word_embedded, last_hidden)

        rnn_output = rnn_output.squeeze(0)
        output = F.log_softmax(self.out(rnn_output), dim=-1)

        return output, hidden


class biEncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(biEncoderGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size//2
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, n_layers, bidirectional=True)
        
        
    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.rnn(embedded, hidden)
        hidden = hidden.reshape(self.n_layers, 1, self.hidden_size * 2)
        return output, hidden

    def init_hidden(self):
        h0 = torch.zeros(self.n_layers * 2, 1, self.hidden_size)
        if USE_CUDA: 
            h0 = h0.cuda()
        return h0



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        hidden = torch.zeros(1, 1, self.hidden_size)
        if USE_CUDA:
            hidden = hidden.cuda()
        return hidden

class Seq2seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size, n_layers):
        super(Seq2seq, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.encoder = EncoderRNN(input_vocab_size, hidden_size, self.n_layers)
        self.decoder = DecoderRNN(output_vocab_size, hidden_size, self.n_layers)

        self.W = nn.Linear(hidden_size, output_vocab_size)
        init.normal_(self.W.weight, 0.0, 0.2)

        self.softmax = nn.Softmax()

    def _forward_encoder(self, x):
        batch_size = x.shape[0]
        init_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(x, init_hidden)
        encoder_hidden_h, encoder_hidden_c = encoder_hidden

        self.decoder_hidden_h = encoder_hidden_h.permute(1,0,2).reshape(batch_size, self.n_layers, self.hidden_size).permute(1,0,2)
        self.decoder_hidden_c = encoder_hidden_c.permute(1,0,2).reshape(batch_size, self.n_layers, self.hidden_size).permute(1,0,2)
        return self.decoder_hidden_h, self.decoder_hidden_c

    def forward_train(self, x, y):
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x)

        H = []
        for i in range(y.shape[1]):
            input = y[:, i]
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (batch_size, vocab_size)
            h = self.W(decoder_output.squeeze(1))
            # h: (batch_size, vocab_size, 1)
            H.append(h.unsqueeze(2))

        # H: (batch_size, vocab_size, seq_len)
        return torch.cat(H, dim=2)

    def forward(self, x):
        decoder_hidden_h, decoder_hidden_c = self._forward_encoder(x)

        current_y = SOS_token
        result = [current_y]
        counter = 0
        while current_y != EOS_token and counter < 100:
            input = torch.tensor([current_y])
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (vocab_size)
            h = self.W(decoder_output.squeeze(1)).squeeze(0)
            y = self.softmax(h)
            _, current_y = torch.max(y, dim=0)
            current_y = current_y.item()
            result.append(current_y)
            counter += 1

        return result