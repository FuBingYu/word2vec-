import os
import re
import time
import string
import random
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import codecs

# Step 1: Data preprocessing.
def Data_Pre(corpus: str, out: str, head = True):
    wf = open(out, 'w', encoding='utf-8')
    sent_list = []
    with open(corpus, encoding='utf-8') as f:
        for line in f:
            l = line.strip()
            l = re.sub("[!\"#$%&\(\)\*\+,-\./:;<=>@\[\]\\\?^_`\{\}~\|]", '', l)   #将所有字符用空格替代
            l = re.sub("\d+\.?\d+", 'NBR', l)  #把所有数字用NBR替代
            sent_list.append(l)
        for sent in sent_list:  #去除文本中的空行
            if sent != '':
                wf.write('{0}\n'.format(sent))
    #print(sent_list)
    wf.close()
    return out
    
raw_file = 'C:/Users/傅冰玉/Desktop/自然语言处理/word2vec进阶版/data/Harry Potter 1.txt'
corpus = Data_Pre(raw_file, 'C:/Users/傅冰玉/Desktop/自然语言处理/word2vec进阶版/data/Harry Potter_.txt')

#读取数据
def read_data(filename: str):
    words = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            l = line.strip().split()
            for word in l:
                words.append(word)
    return words

words = read_data(corpus)
print(words)
print('Data size: {0} words.'.format(format(len(words), ',')))

# Step 2: Build the dictionary and replace rare words
def build_dataset(words, vocabulary_size=40000):
    token_count = []
    token_count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    word2idx = dict()
    data = []
    for word, _ in token_count:
        word2idx[word] = len(word2idx)
    word_set = set(word2idx.keys())
    for word in words:
        if word in word_set:
            index = word2idx[word]
        else:
            index = 0
        data.append(index)
    idx2word = {idx: word for word, idx in word2idx.items()}
    return data, token_count, word2idx, idx2word

vocabulary_size = 40000
data, count, word2idx, idx2word = build_dataset(words, vocabulary_size)
words = list(word2idx.keys())
print('Most common words ', count)
print('Sample data: index: {0}, token: {1}'.format(data[:30], [idx2word[i] for i in data[:30] ]))

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(data, batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
      buffer.append(data[data_index])
      data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
      target = skip_window
      targets_to_avoid = [skip_window]
      for j in range(num_skips):
          while target in targets_to_avoid:
              target = random.randint(0, span - 1)
          targets_to_avoid.append(target)
          batch[i * num_skips + j] = buffer[skip_window]
          labels[i * num_skips + j, 0] = buffer[target]
      buffer.append(data[data_index])
      data_index = (data_index + 1) % len(data)
  return batch, labels

data_index = 0
batch_size = 16
skip_window = 4
num_skips = 8
batch, labels = generate_batch(data=data, batch_size=batch_size, num_skips=num_skips, skip_window=skip_window)

for i in range(16):
  print(batch[i], idx2word[batch[i]],'->', labels[i, 0], idx2word[labels[i, 0]])

# Step 4: Build a skip-gram model.
class SkipGram(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vocabulary_size = args.vocabulary_size
        self.embedding_size = args.embedding_size
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size) # W = vd lookup  [1*v']*[V*embedding_size]  -> [v* embedding_size]
        self.output = nn.Linear(self.embedding_size, self.vocabulary_size) # 输出层
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):    #x 16
        x = self.embedding(x)  #x  16*128
        x = self.output(x)     #x  16*40000
        log_ps = self.log_softmax(x)  #x 16*40000
        return log_ps

# Step 5: Begin training.
class config():
    def __init__(self):
        self.num_steps = 1000
        self.batch_size = 128
        self.check_step = 20
        self.vocabulary_size = 40000
        self.embedding_size = 200  # Dimension of the embedding vector.
        self.skip_window = 4  # How many words to consider left and right.
        self.num_skips = 8  # How many times to reuse an input to generate a label.
        self.use_cuda = torch.cuda.is_available()
        self.lr = 0.03

args = config()
model = SkipGram(args)
print(model)

if args.use_cuda:
    model = model.to('cuda')

nll_loss = nn.NLLLoss()
adam_optimizer = optim.Adam(model.parameters(), lr=args.lr)

print('-'*50)
print('Start training.')
average_loss = 0
start_time = time.time()
for step in range(1, args.num_steps):
    batch_inputs, batch_labels = generate_batch(
        data, args.batch_size, args.num_skips, args.skip_window)
    batch_labels = batch_labels.squeeze()
    batch_inputs, batch_labels = torch.LongTensor(batch_inputs), torch.LongTensor(batch_labels)
    if args.use_cuda:
        batch_inputs, batch_labels = batch_inputs.to('cuda'), batch_labels.to('cuda')
    log_ps = model(batch_inputs)
    loss = nll_loss(log_ps, batch_labels)
    average_loss += loss
    adam_optimizer.zero_grad()
    loss.backward()
    adam_optimizer.step()
    if step % args.check_step == 0:
        end_time = time.time()
        average_loss /= args.check_step
        print('Average loss as step {0}: {1:.2f}, cost: {2:.2f}s.'.format(step, average_loss, end_time-start_time))
        start_time = time.time()
        average_loss = 0
print('Training Done.')
print('-'*50)

final_embedding = model.embedding.weight.data#权重
print(final_embedding.shape)

#step 6:Begin Visualing
# 在画图的时候，可以去除停用词
stopword_file = codecs.open("C:/Users/傅冰玉/Desktop//自然语言处理/word2vec进阶版/data/teststopwords.txt", "r", encoding='utf-8')
word_list = []
for word in stopword_file.readlines():
    word_list.extend(word.strip().split(' '))
stopwords = set(word_list)
# 取词频前500个词
labels = [ idx2word[i] for i in range(500) ]
new_words = []
for i  in range(len(labels)):
    if re.findall(re.compile("[A-Z][a-z]+"),labels[i])!= []:
        new_words.append(labels[i])
second_new_words = []
for i in range(len(new_words)):
    if new_words[i] in stopwords:
        continue
    else:
        second_new_words.append(new_words[i])

# 用TSNE降维
matplotlib.use("Agg")
# n_components：嵌入式空间的维度，perplexity：浮点型，
# init：嵌入的初始化（ PCA初始化不能用于预先计算的距离，并且通常比随机初始化更全局稳定。），
# n_iter：优化时的最大迭代次数
tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=5000)
low_dim_embs = tsne.fit_transform(final_embedding[:500, :])  # 只画词频前200的词语

# 画图
print('Start Visualing.')
plt.figure(figsize=(18, 18))
for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    #if(x>-25 and x<25) and (y<25 and y>-25):
    if x<400 and x>-400:
        plt.scatter(x, y)
        plt.annotate(label,xy=(x, y),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
plt.savefig('tsne.png')

print('Visualing done.')
