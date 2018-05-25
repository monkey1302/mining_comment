from gensim.models import word2vec  
import logging

# 主程序  
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)  
sentences =word2vec.Text8Corpus("./corpus.txt")  # 加载语料  
model =word2vec.Word2Vec(sentences, size=200)  #训练skip-gram模型，默认window=5
model.save("mining.model") #最终就是要这个文件
#model =word2vec.Word2Vec.load("mining.model")  #下次使用的时候直接加载，不用再训练
 

# -- 测试训练效果 计算与这个词相似度最高的top20 -- 
try:  
    y1 = model.most_similar("好评",topn=20)
    for item in y1:  
        print (item[0], item[1] )

except KeyError:  
    y1 = 0  



