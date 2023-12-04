# -*- coding:utf-8 _*-
import torch
import torch.nn as nn
from transformers import MBart50Tokenizer
from transformers import MBartForConditionalGeneration
import sentencepiece_model_pb2 as sent_model
from tokenization_mlm import MLMTokenizer

# Load tokenizer and model
tokenizer = MLMTokenizer.from_pretrained('checkpoints/facebook-mbart-large-50')
model = MBartForConditionalGeneration.from_pretrained('checkpoints/facebook-mbart-large-50')
model.resize_token_embeddings(len(tokenizer))

# Configuration and weight initialization
std = model.config.init_std
weight = model.model.shared.weight.data.clone()
all_vocab = [k for k in tokenizer.get_vocab().keys()]

# Reading and updating the vocabulary based on training data
vocab = all_vocab.copy()[:50]
for lang in ['de_DE','en_XX','it_IT','nl_XX']:
    for mode in ['gold', 'silver', 'bronze']:
        for i in range(2):
            print(lang, mode, i)
            with open('./data/{}/{}/train.{}'.format(lang[:2], mode, str(i)), 'r') as f:
                for line in f.readlines():
                    vocab.extend(tokenizer.tokenize(line.strip()))
print(len(set(vocab)))
vocab = list(set(all_vocab) & set(vocab))
print(len(vocab))

# Load and update the SentencePiece model
m = sent_model.ModelProto()
m.ParseFromString(open('checkpoints/facebook-mbart-large-50/sentencepiece.bpe.model', 'rb').read())

# Update the model's embeddings based on the new vocabulary
cur_id = 0
for i in range(len(tokenizer)):
    if all_vocab[i] in vocab:
        vocab.pop(vocab.index(all_vocab[i]))
    else:
        id = all_vocab.index(vocab[0])
        score = m.pieces[id-1].score
        m.pieces[i - 1].piece = vocab[0]
        m.pieces[i - 1].score = score
        model.model.shared.weight.data[i, :] = weight[id, :]
        vocab.pop(0)
    cur_id += 1
    if len(vocab) == 0:
        break
for i in range(cur_id - 1, len(m.pieces)):
    m.pieces.pop(-1)

# Adjust the model's shared weights
model.model.shared.weight.data[cur_id: cur_id + 54, :] = model.model.shared.weight.data[-54:, :]
model.resize_token_embeddings(cur_id + 54)
print(cur_id + 54)

# Save the modified model and SentencePiece model
model.save_pretrained('.checkpoints/mbart-large-50/')
with open('checkpoints/mbart-large-50/sentencepiece.bpe.model', 'wb') as f:
    f.write(m.SerializeToString())
