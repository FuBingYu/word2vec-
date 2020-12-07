import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM


#加载预训练好的模型、分词器
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def bert_sen_training(str):
    #处理数据
    sen = str
    marked_text = "[CLS] " + sen + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    #['[CLS]', 'harry', 'potter', 'and', 'the', 'sorcerer', "'", 's', 'stone', '[SEP]']
    length = len(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    #print(indexed_tokens)
    # [101, 4302, 10693, 1998, 1996, 23324, 1005, 1055, 2962, 102]
    segments_ids = [1] * len(tokenized_text)
    # 对编码进行转换，以便输入Tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # 开始测试模型
    model.eval()
    with torch.no_grad():
        encoded_layers,_ = model(tokens_tensor, segments_tensors)
    #for tup in zip(tokenized_text, encoded_layers[0]):
        #print(tup)

    #求句向量、词向量
    sentence_embedding = torch.mean(encoded_layers[0], 1)
    sentence_embedding = sentence_embedding.numpy()      # 把tensor张量转换为数组
    sentence_embedding = sentence_embedding[1:length-1]
    tokenized_text = tokenized_text[1:length-1]
    tokenized_text_dic =dict(zip(tokenized_text, sentence_embedding))
    #print(tokenized_text_dic)
    '''{'harry': -0.014254867, 'potter': -0.021509541, 'and': -0.01615799, 'the': -0.020105276,
    'sorcerer': -0.020037206, "'": -0.017196726, 's': -0.021674259, 'stone': -0.018208722}'''
    sentence_embedding = torch.from_numpy(sentence_embedding)
    return sentence_embedding

    #h = tokenized_text_dic.get("harry")
    #  print(h)  -0.014254867


def bert_word_training(word):
    token_input = tokenizer(word, return_tensors='pt')
    token_embedding, _ = model(**token_input)
    word_embedding = token_embedding[0][1]
    return word_embedding


s1 = bert_sen_training("Harry Potter and the Sorcerer's Stone ")
s2 =bert_sen_training("Harry Potter and the Deathly Hallows")

w1 = bert_word_training("Harry")
w2 = bert_word_training("Ron")
# 利用已训练好的模型，查看词之间或句子之间的相似度
def Cal_sim(t1,t2):

    n = torch.dot(t1, t2)
    m = torch.sqrt(torch.sum(torch.pow(t1, 2)))*torch.sqrt(torch.sum(torch.pow(t2, 2)))
    return n/m

sen_sim = Cal_sim(s1,s2)
print("句子间的相似度",sen_sim)
word_sim = Cal_sim(w1,w2)
print("词语间的相似度",word_sim)
#tensor(0.8624, grad_fn=<DivBackward0>)
