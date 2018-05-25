# -*- coding: utf-8 -*-

# ----- 这里代码主要进行名词提取、每个名词的修饰词提取 ------

import time
from scipy import stats
import math
import numpy as np
from sklearn.cluster import KMeans
import os
from pyltp import Segmentor
from pyltp import Postagger
import json
from gensim.models import word2vec

LTP_DATA_DIR = './ltp_data'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')


segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型


stop_word = [line.strip() for line in open('stop_word.txt','r',encoding='utf-8').readlines()]
model =word2vec.Word2Vec.load("./mining.model")#之前用所有评论训练好的模型



    
def readcomment(comment_path):

    '''
    input: one comment file path
    output: comment list ['comment1','comment2',....]
    '''
    # -- 读文件的每一条记录 --
    with open(comment_path,'r',encoding='utf-8') as f:
            data = f.read().strip().split("\n")
                  
    
    # -- 删除错误行 --
    wrong_index = []
    for i in range(len(data)):
        data[i] = json.loads(data[i])
        if data[i].__contains__("sku_id"):        
            wrong_index.append(i)
    for i in range(len(wrong_index)):
        ind = wrong_index[i]-i
        del data[ind]

    # -- 从每条记录中，提取出评论正文 --
    comments = []
    for item in data:
        if item['content']!= None:
            comments.append( item['content'])        
    return (comments)





def choose_adj(adj_list):  #方法1，选取频率最高的adj
    '''
    input:[adj1,adj2, ...] 一个形容词的列表，有重复的
    output: adj 一个形容词，词频最高的
    '''
    adj_count = {}
    for adj in adj_list:
        if adj_count.__contains__(adj):
            adj_count[adj] +=1
        else:
            adj_count[adj] = 1
    res=sorted(adj_count.items(),key = lambda x:x[1],reverse = True) #[(adj1,count1) , (adj2,count2) ......]
    return res[0]






def w2v_kmeans(full_list,k):

    '''
    input: full_list：[ [noun,count,[adj_list]] , [] ,[]...] , [ [名词1，词频，[形容词列表]], [], []...]
    input: k：Kmeans聚类时，k的值
    output : 无，结果会写到文件中
    '''

    noun_list = [item[0] for item in full_list] #名词列表
    

    # -- 亲属关系的词,过滤掉 --
    filtering_word = ['宝宝','时候', '孩子','图片', '学生','奶奶','公公','好评',"棒棒",'小哥','东西','产品','小孩', '小孩子', '小朋友', '小宝宝', '婴儿', '朋友', '同事', '亲戚', '邻居','妈妈', '父母', '老妈', '爸妈', '老人', '妹妹', '爸爸', '姐姐', '弟弟','家人', '儿子', '老婆', '家里人', '老公', '女儿']

    # --- word2vec part ---

    global model
    vec_list = [] #每个词的向量

    
    # -- 对每个名词，找到它词向量，不是所有词都有对应的向量，因此要把这些没有词向量的名词整条记录删除 --
    wrong_index=[]
    for i in range(len(noun_list)):
       
        try:
            vec_list.append(model[noun_list[i]]) #把词向量记录下来
        except Exception:
            wrong_index.append(i)
    for i in range(len(wrong_index)):
        index = wrong_index[i]-i
        del noun_list[index]
        del full_list[index]
       

    # -- 删除掉不能向量化的名词之后，再获词频列表，和形容词列表 --
    adj_list = [item[2] for item in full_list]
    count_list = [item[1] for item in full_list]
    

    total_count = sum(count_list)
    

    # -- k-mean part --
    estimator = KMeans(n_clusters=k,max_iter=100000) #初始化
    cluster_result = estimator.fit_transform(vec_list)#开始聚类
    
    data_len = len(cluster_result)
    label_pred = estimator.labels_ #获取聚类标签

    
    f = open("noun_adj.txt","w+",encoding='utf-8') #把最终结果写在文件里

    # -- 处理结果 --
    cluster_count = {} #每类计数
    score_dict = {}  #每类每个点的分值 {1:[[index,分值],[index,分值]...], 2:[..], 3:[...]}
    #初始化两个字典
    for i in range(k):
        cluster_count[str(i)] = 0
        score_dict[str(i)] = []
    
    #每类计数
    for label in label_pred:
        cluster_count[str(label)] +=1

    #每类每个点算分值 返回[第几类，分值]
    for i in range(data_len):
        label = label_pred[i]
        distance = cluster_result[i][label]
        score = 0*math.exp(-distance)+1*(count_list[i]/total_count) #选取代表词，可以调整参数，第一项是距中心点距离，第二项是词频
        score_dict[str(label)].append([i,score])

    res_index = []
    for i in range(k):# 循环每个类   
            
        res = sorted(score_dict[str(i)], key = lambda x :x[1],reverse=True) #获取这一类中所有的名词的index 
        
        #f.write("\n------cluster:{}-----\n".format(i))
        print("------cluster:{}----".format(i))

          
        word_index = [] #名词的index
        for item in res:
            word_index.append(item[0])
                        
        words = []
        for index in word_index:
            words.append(noun_list[index])


        if words[0] in filtering_word:  #过滤掉字典中的
                
                print("this cluster will be dropped")
                continue

        if count_list[word_index[0]]<total_count/200: #某一类中，词频最高的词，它的词频小于总词数的1/200,过滤掉这个词所在的类
            print(words)
            print("this cluster will be dropped")
            continue

        res_index.append(res[0][0]) #第一个元素的index
                
        print(words)
        #f.write(str(words))

    result = [noun_list[i] for i in range(len(noun_list)) if i in res_index]
    print("---------final result-------")
    f.write("\n---------final result-------\n")
    f.write(str(result)+"\n")
   
    print(result)



    #------给每个result noun找到一个adj-----
    for i in res_index:
        this_adj_list = adj_list[i]
        final_adj = choose_adj1(this_adj_list) 
        print (noun_list[i],final_adj)
        f.write(str(noun_list[i])+"\t"+str(final_adj)+"\n")
    f.close()





def find_adj(words_list,postags_list,i):
    '''
    作用，给定一句话，以及名词的位置，给这个名词找到修饰它的形容词
    input: [word1, word2, ...] 词的列表
    input: [verb, noun, adj ...]每个词对应的词性列表
    input: i， 当前名词的index
    output:[adj1,adj2,...] 修饰这个名词的所有形容词列表
    '''
    adj = []
    
    # -- 向后找 --
    if i < len(words_list)-1:
        if i <len(words_list)-2:#不是倒数第二个词
            if postags_list[i+1] == 'wp': #如果遇到标点，就不向后了
                pass
            else:
                if postags_list[i+1] in ['a']:
                    adj.append(words_list[i+1])
                elif postags_list[i+1] in ['d','v']:
                    if postags_list[i+2] in ['a']:
                        adj.append(words_list[i+1]+words_list[i+2])
                        
        else: #是倒数第二个词，只向后找一个
            if postags_list[i+1] in ['a']:
                adj.append(words_list[i+1])
                
    #print(words_list[i])
    #print(adj)

    # -- 向前找 --
    if i>0: 
        if i >1: #不是正数第二个词
            if postags_list[i-1] == 'wp': #如果遇到标点，就不向前了
                pass
            else:
                if postags_list[i-1] in ['a']:
                    adj.append(words_list[i-1])
                elif postags_list[i-1] in ['d','v']:
                    if postags_list[i-2] in ['a']:
                        adj.append(words_list[i-2]+words_list[i-1])
        else:#是正数第二个词
            if postags_list[i-1] in ['a']:
                adj.append(words_list[i-1])

    return adj

def mining(comments):

    '''
    input : [comment1, comment2, ...]
    output: 无
    '''
    word_count ={} #{noun1:count , noun2：count, ....}
    total_wordcount= 0
    noun_adj = {} #{noun:[adj_list], noun:[adj_list].....}
    
    for comment in comments:   
        words = segmentor.segment(comment)  # 分词 
        words_list = [word.lower() for word in list(words) if word not in stop_word] #过滤掉停词
        words_list = [word.replace('.','').replace(',','').replace('?','').replace('!','').replace('@','') for word in words_list] #去除英文标点
        while '' in words_list: #删除空单词
            words_list.remove('')
        total_wordcount +=len(words_list) #累加词的数量

        
        postags = postagger.postag(words_list)  # 词性标注
        postags_list = list(postags)

        
        
        # -- 给名词附加上形容词 --
        for i in range(len(words_list)):
            
            if postags_list[i]=='n':
                adj_list=find_adj(words_list,postags_list,i)
                if noun_adj.__contains__(words_list[i]):
                    noun_adj[words_list[i]].extend(adj_list)
                else:
                    noun_adj[words_list[i]]=adj_list
                
          
        # -- 名词计数 --
        for i in range(len(words_list)):
            if postags[i]=="n" and len(words_list[i])>1:
                word = words_list[i]        
                if word_count.__contains__(word):
                    word_count[word]=word_count[word]+1
                else:
                    word_count[word] = 1
                    
    # -- 按词频排序 --
    result = sorted(word_count.items(),key = lambda x:x[1],reverse = True) #list:[(noun1,count1),(noun2,count2),.....]
    

    full_list=[] #关于所有名词的列表，包含词、词频、形容词列表 [[word1, 10 ,[adj1,adj2]], ...]
    for item in result:
        word = item[0]
        count = item[1]
        adj=noun_adj[word]
        full_list.append([word,count,adj])


    # -- 设置k-means的k值，目标是尽量设大一些，为了防止后面聚类不够细 --  
    lens = len(full_list)
    if lens > 50:
        k=50
    else:
        k=int(len*0.7)
    
    w2v_kmeans(full_list,k) #第二个参数是聚类的个数



        



# -- 完整部分，循环读所有文件 ---
'''
folder_path = "./raw_comments/"
files= os.listdir(folder_path) #得到文件夹下的所有文件名称
f = open("aspect_result1.csv","w+",encoding='utf-8')
for file in files:
    
    comment_path = folder_path+file

    comments = readcomment(comment_path)
    if comments == None:
        result = " "
    else:     
        result = mining(comments)

'''
# -- 测试部分，只读一个文件 --



comment_path = "./raw_comments/item_comments_jd_1741527728"
comments = readcomment(comment_path) # [comment1, comment2, ...]

if comments == None:
    result = " "
else:     
    result = mining(comments)




segmentor.release()  # 释放模型
postagger.release()  # 释放模型

