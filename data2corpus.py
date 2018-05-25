# -*- coding: utf-8 -*-

import json
import os

from pyltp import SentenceSplitter
import os
from pyltp import Segmentor
from pyltp import Postagger


LTP_DATA_DIR = './ltp_data'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')


segmentor = Segmentor()  #初始化实例
segmentor.load(cws_model_path) #加载实例
postagger = Postagger() 
postagger.load(pos_model_path)  

def load_and_split(comment_path):

    # -- 读入评论 --
    with open(comment_path,'r',encoding='utf-8') as f:
        data = f.read().strip().split("\n")
        
    # data的格式 [{},{},...{}]

        
    # -- 删除错误行 --
    wrong_index = []
    for i in range(len(data)):
        data[i] = json.loads(data[i])
        if data[i].__contains__("sku_id"):
            
            wrong_index.append(i)
    for i in range(len(wrong_index)):
        ind = wrong_index[i]-i
        del data[ind]

    if len(data)==0:
        return "empty"
    


    # -- NLP分词 写入文件--
    for item in data:
        comment = item['content']
        if comment=="":
            continue
        words = segmentor.segment(comment)
        words = list(words)
        words = [word.replace('.','').replace(',','').replace('?','').replace('!','').replace('@','').lower() for word in words]
        while '' in words:
            words.remove('')
            
        postags = postagger.postag(words)
        postags = list(postags)

        corpus_list = [words[i] for i in range(len(words)) if postags[i]!='wp']
        output.write(" ".join(corpus_list)+"\n")
  

    


folder_path = "./raw_comments/"
files= os.listdir(folder_path)
lens = len(files)
output = open('corpus.txt', 'w+',encoding = 'utf-8')

for i in range(lens):
    comment_path = folder_path+files[i]
    load_and_split(comment_path)
    
    
output.close()


segmentor.release()
postagger.release()


