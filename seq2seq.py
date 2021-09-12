import random
import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score

from tqdm import tqdm
import os
from nltk.translate.bleu_score import corpus_bleu

from model import AttnDecoderRNN, EncoderRNN, DecoderRNN, biEncoderGRU
from config import *
from utils import time_since, show_plot
from lang import prepare_data, normalize_string, variables_from_pair, variable_from_sentence



input_lang, output_lang, pairs = prepare_data('cn', 'eng', False)

# Print an example pair
print(random.choice(pairs))

teacher_forcing_ratio = 0.5
clip = 5.0

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    if MODEL_NAME in ['rnn', 'gru']:
        encoder_hidden = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    elif MODEL_NAME == 'lstm':
        encoder_hidden = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    else:
        encoder_hidden = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    # Prepare input and output variables
    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di] # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            
            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = torch.LongTensor([[ni]]) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length

def train_attn(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size).cuda() # SOS, EOS

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]]).cuda()

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def evaluate(sentence, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    
    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([[SOS_token]]) # SOS
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni.item()])
            
        # Next input is chosen word
        decoder_input = torch.LongTensor([[ni]])
        if USE_CUDA: decoder_input = decoder_input.cuda()
    
    return decoded_words

# def test():
#     fin = open('./test.txt', 'r')
#     fout = open('./result_{}.txt'.format(MODEL_NAME), 'w')
#     for line in tqdm(fin.readlines()):
#         output_words = evaluate(normalize_string(line.strip()))
#         fout.write(' '.join(output_words) + '\n')
    
#     candidates = open('./result_{}.txt'.format(MODEL_NAME), 'r').readlines()
#     references = open('./cn-eng_eng.txt', 'r').readlines()
#     print(len(candidates), len(references))
#     for i in range(len(candidates)):
#         candidates[i] = normalize_string(candidates[i].strip()).split()
#         references[i] = [normalize_string(references[i].strip()).split()]
#     score = bleu_score(candidates, references)
#     print(score)

# def evaluate_full():
#     fin = open('./data/cn-eng.txt', 'r')
#     # fout = open('./evaluate_result_{}.txt'.format(MODEL_NAME), 'w')
#     candidates = []
#     references = []
#     for line in tqdm(fin.readlines()):
#         output_words = evaluate(normalize_string(line.strip().split('\t')[0]))
#         candidates.append(' '.join(output_words))
#         references.append(line.strip().split('\t')[1])
#     # for line in res:
#     #     fout.write(line)
#     print(len(candidates), len(references))
#     for i in range(len(candidates)):
#         candidates[i] = normalize_string(candidates[i].strip()).split()
#         references[i] = [normalize_string(references[i].strip()).split()]
#     score = bleu_score(candidates, references)
#     print(score)

def valid():
    fin = open('./data/cn-eng_valid.txt', 'r')
    candidates = []
    references = []
    for line in tqdm(fin.readlines()):
        output_words = evaluate(normalize_string(line.strip().split('\t')[0]))
        if output_words[-1] == '<EOS>':
            output_words = output_words[:-1]
        candidates.append(' '.join(output_words)) # remove EOS
        references.append(line.strip().split('\t')[1])
    print(len(candidates), len(references))
    for i in range(len(candidates)):
        candidates[i] = normalize_string(candidates[i].strip()).split()
        references[i] = [normalize_string(references[i].strip()).split()]
    score = bleu_score(candidates, references)
    print(score)

def evaluate_randomly():
    pair = random.choice(pairs)
    
    output_words = evaluate(pair[0])
    output_sentence = ' '.join(output_words)
    
    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')



# Initialize models
if MODEL_NAME == 'bigru':
    encoder = biEncoderGRU(input_lang.n_words, hidden_size, n_layers)
    decoder = DecoderRNN(hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p, model_name='gru')
elif MODEL_NAME == 'attn':
    encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers, model_name='gru')
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=dropout_p, max_length=MAX_LENGTH)
else:
    encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers, model_name=MODEL_NAME)
    decoder = DecoderRNN(hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p, model_name=MODEL_NAME)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()



# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

# # Begin!
for epoch in range(1, n_epochs + 1):
    
    # Get training data for this cycle
    training_pair = variables_from_pair(random.choice(pairs), input_lang, output_lang)
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    # Run the train function
    if MODEL_NAME == 'attn':
        loss = train_attn(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    else:
        loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch == 0: continue

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
torch.save(encoder.state_dict(), os.path.join(SAVE_DIR, 'encoder_'+MODEL_NAME+MODEL_ADD_NAME))
torch.save(decoder.state_dict(), os.path.join(SAVE_DIR, 'decoder_'+MODEL_NAME+MODEL_ADD_NAME))

# encoder.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'encoder_'+MODEL_NAME+MODEL_ADD_NAME)))
# decoder.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'decoder_'+MODEL_NAME+MODEL_ADD_NAME)))

show_plot(plot_losses)
evaluate_randomly()
valid()
# test()