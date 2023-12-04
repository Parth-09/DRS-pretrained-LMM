# -*- coding:utf-8 _*-
import torch
import torch.nn as nn
from transformers import MBart50Tokenizer
from transformers import MBartForConditionalGeneration
import sentencepiece_model_pb2 as sent_model
from tokenization_mlm import MLMTokenizer

# Loading the tokenizer and model from the specified checkpoint
tokenizer = MLMTokenizer.from_pretrained('checkpoints/facebook-mbart-large-50')
model = MBartForConditionalGeneration.from_pretrained('checkpoints/facebook-mbart-large-50')

# Resizing the token embeddings of the model to match the tokenizer's length
model.resize_token_embeddings(len(tokenizer))

# Storing the standard deviation for initialization and cloning the model's weights
std = model.config.init_std
weight = model.model.shared.weight.data.clone()
# Extracting all vocabulary keys from the tokenizer
all_vocab = [k for k in tokenizer.get_vocab().keys()]

# Getting the size of the token embeddings
num_tokens, embedding_dim = model.model.shared.weight.size()

# Initial vocabulary update based on training text data
vocab = all_vocab.copy()[:50]
for lang in ['de_DE','en_XX','it_IT','nl_XX']:
    for mode in ['gold', 'silver', 'bronze']:
        for i in range(2):
            print(lang, mode, i)
            # Reading the training data and extending the vocabulary with tokenized words
            with open('./data/{}/{}/train.{}'.format(lang[:2], mode, str(i)), 'r') as f:
                for line in f.readlines():
                    vocab.extend(tokenizer.tokenize(line.strip()))
print(len(set(vocab)))
# Filtering the vocabulary to include only those in all_vocab
vocab = list(set(all_vocab) & set(vocab))
print(len(vocab))

# Loading the SentencePiece model
m = sent_model.ModelProto()
m.ParseFromString(open('checkpoints/facebook-mbart-large-50/sentencepiece.bpe.model', 'rb').read())

# Updating the model's embeddings based on the new vocabulary
cur_id = 0
for i in range(len(tokenizer)):
    if all_vocab[i] in vocab:
        vocab.pop(vocab.index(all_vocab[i]))
    else:
        id = all_vocab.index(vocab[0])
        score = m.pieces[id-1].score
        # Updating the SentencePiece model
        m.pieces[i - 1].piece = vocab[0]
        m.pieces[i - 1].score = score
        # Updating the model's weights
        model.model.shared.weight.data[i, :] = weight[id, :]
        vocab.pop(0)
    cur_id += 1
    if len(vocab) == 0:
        break
# Cleaning up any remaining SentencePiece model pieces
for i in range(cur_id - 1, len(m.pieces)):
    m.pieces.pop(-1)

# Adjusting the shared weight data of the model
model.model.shared.weight.data[cur_id: cur_id + 54, :] = model.model.shared.weight.data[-54:, :]
# Resizing the token embeddings after updates
model.resize_token_embeddings(cur_id + 54)
print(cur_id + 54)

# Saving the modified model and SentencePiece model
model.save_pretrained('.checkpoints/mbart-large-50/')
with open('checkpoints/mbart-large-50/sentencepiece.bpe.model', 'wb') as f:
    f.write(m.SerializeToString())
