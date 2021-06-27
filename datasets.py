!pip install transformers
import json
import os
import re
import string
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_path = keras.utils.get_file("train.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json")
eval_path = keras.utils.get_file("eval.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")
with open(train_path) as f: raw_train_data = json.load(f)
with open(eval_path) as f: raw_eval_data = json.load(f)
max_seq_length = 384

print("Data = ", raw_train_data)

# Knowledge about Montana

first_one = raw_train_data['data'][2]['paragraphs'] 

for index, elem in enumerate(first_one):
  print(index, elem['context'])

first_one = raw_train_data['data'][0]['paragraphs'][0]['qas'][0] 
print(first_one)
print(type(first_one))


print('_____________________________________________\n\n')

data = raw_train_data['data'][0]['paragraphs']

print(type(data))
for index, e in enumerate(data):
  print(index, e)
  print(index, type(e))

print('_____________________________________________\n\n')


print('print1 = ', type(raw_train_data['data'][0]['paragraphs'][1]['qas'][0]))
print('print2 = ', raw_train_data['data'][0]['paragraphs'][1]['qas'][0].keys())
print('print3 = ', raw_train_data['data'][0]['paragraphs'][1]['qas'][0]['answers'])

print('_____________________________________________\n\n')
for e in raw_train_data['data'][1]['paragraphs'][5]['qas']:
  print(e)