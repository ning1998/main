import math
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def average_position(predict,real):#真实并发症的平均预测位置排名
    a=0
    for item in real:
        if item in predict:
            a+=np.where(predict==item)[0][0]
    a=a/len(real)
    return a

def P_k(k,predict,real):
    P=0
    for item in predict[:k]:
        if item in real:
            P+=1
    P=P/k
    return P
def R_k(k,predict,real):
    R=0
    for item in real:
        if item in predict[:k]:
            R+=1
    R=R/len(real)
    return R
def MAP_1(predict,real):#for one dis
    MAP=0
    for r in real:
        if r not in predict:
            continue
        ri=np.where(predict==r)[0][0]+1
        MAP+=P_k(ri,predict,real)
    MAP=MAP/len(real)
    return MAP

def load_embedding(path='./data/embedding.txt'):
    #在word2id中保存的是所有实体，但是embedding中不包含没有出入边的实体，
    #id2embedkey对应word2vec中的id，value对应相应id实体向量
    f=open(path,'r')
    lines=f.readlines()
    f.close()
    id2embed={}
    num_entity=int(lines[0].replace('\n','').split()[0])
    embed_dim=int(lines[0].replace('\n','').split()[1])
    print(num_entity==len(lines)-1)
    
    for i in range(1,len(lines)):
        line=lines[i]
        S=line.replace('\n','').split()
        if len(S)!=embed_dim+1:
            print('err11')
        id2embed[int(S[0])]=np.zeros((embed_dim))
        for j in range(0,embed_dim):
            id2embed[int(S[0])][j]=float(S[j+1])
    print('load_embedding over')
    return id2embed


def load_entity2id():
    f=open('../data/entity2id.txt','r',encoding='utf-8')
    lines=f.readlines()
    f.close()
    entity2id={}
    id2entity={}
    for line in lines:
        line=line.replace('\n','').split('\t')
        if len(line)!=2:
            print("load entity2id false")
            return
        entity2id[line[0]]=int(line[1])
        id2entity[int(line[1])]=line[0]
    return entity2id,len(entity2id)

def load_dis_acc(path,entity2id):
    #加载500个疾病，将其与所有疾病匹配，判断是否为并发，并计算MAP
    f=open(path,'r',encoding='utf-8')
    rels=f.readlines()
    f.close()
    dis2acc={}
    for r in rels:
        rel=r.replace('\n','').split('\t')
        if len(rel)!=3:
            print(rel)
        if rel[2]=='accompany':
            if entity2id[rel[0]] not in dis2acc:
                dis2acc[entity2id[rel[0]]]=[]
            if entity2id[rel[1]] not in dis2acc[entity2id[rel[0]]]:
                dis2acc[entity2id[rel[0]]].append(entity2id[rel[1]])

    print('load_dis_acc over')
    return dis2acc

def load_evaluate_data(idx,entity2id,id2embed ,dis2acc):
    head=[]
    tails=[]
    real_acc=[]#要evaluate的疾病的真正并发症

    all_evaluate_dis=[]
    all_dis=list(set(dis2acc.keys()))
    for dis in dis2acc:
        all_evaluate_dis.append(dis)
        if len(all_evaluate_dis)==500:
            break
    
    all_X=[]#长度与real_acc一样
    for dis1_id in all_evaluate_dis[idx:idx+1]:
        X=[]
        this_tail=[]
        head.append(dis1_id)
        real_acc.append(dis2acc[dis1_id])
        for dis2_id in all_dis:
            if dis2_id not in id2embed:
                continue
            this_X=[id2embed[dis1_id],id2embed[dis2_id]]
            X.append(this_X)
            this_tail.append(dis2_id)
        tails.append(this_tail)
        all_X.append(X)
    print('load_evaluate_data over')
    return all_X,real_acc,head,tails



num_rel=2
entity2id,num_entity=load_entity2id()
dis2acc=load_dis_acc('../data/train.txt',entity2id)
id2embed=load_embedding()


from keras.optimizers import Adagrad, Adam, SGD, RMSprop, Nadam
from keras.callbacks import EarlyStopping
from my_model import TextCNN
print('Build model...')
model = TextCNN(embedding_dim=100,class_num=2,dis_embedding=128).get_model()
model.load_weights('./model/my_model')
model.compile(optimizer = SGD(lr = 0.001, decay = 1e-4),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
#model.summary()

MAP=0
P=0
R=0
a=0
k=10
for i in range(0,500):
    print(i)
    all_X,real_acc,head,tails=load_evaluate_data(i,entity2id,id2embed,dis2acc)
    print(np.array(all_X).shape)
    print('Evaluate...')
    this_tails=np.array(tails[0])#是id形式
    this_real_acc=np.array(real_acc[0])#是id形式
    eva_data=np.array(all_X[0])
    this_result = model.predict(eva_data)
    
    this_result=this_result[:,1]#判断是并发的概率
    this_result=this_tails[np.argsort(-this_result)]#按照并发可能性排序，从高到低
    

    '''
    this_result=this_result.argmax(axis=1)#0或1
    #print(this_result[0:20])
    this_result=this_tails[np.where(this_result==1)[0]]
    print(this_result.shape)
    print(this_real_acc)
    #print(this_tails[0:20])
    '''
    MAP+=MAP_1(this_result,this_real_acc)
    P+=P_k(k,this_result,this_real_acc)
    R+=R_k(k,this_result,this_real_acc)
    a+=average_position(this_result,this_real_acc)
P=P/500
R=R/500
MAP=MAP/500
a/=50
print("MAP: ",MAP)
print("P_K: ",P)
print("R_K: ",R)
print("average_position of real acc: ",a)
'''
Evaluate...
MAP:  0.25303720086161947
P_K:  0.10400000000000034
R_K:  0.4575576062826062
average_position of real acc:  54.23829193584191
'''
