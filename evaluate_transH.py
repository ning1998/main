import math
import numpy as np
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
    print(predict[0:k])
    print(real[0:k])
          
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

def load_embedding(num_rel,num_entity):
    A=np.zeros((num_rel,100))
    rel_vec=np.zeros((num_rel,100))
    entity_vec=np.zeros((num_entity,100))
    f=open('./transH/A.txtbern','r',encoding='utf-8')
    lines=f.readlines()
    f.close()
    print(len(lines))
    for i in range(0,len(lines)):
        line=lines[i].replace(' \n','')
        line=line.split(' ')
        if len(line)!=100:
            print(len(line))
            print(line)
            print("Load A error")
            return
        for j in range(0,100):
            A[i][j]=float(line[j])
    
    f=open('./transH/relation2vec.txtbern','r',encoding='utf-8')
    lines=f.readlines()
    f.close()
    if len(lines)!=num_rel:
        print('len(lines)!=rel_num')
    for i in range(0,num_rel):
        line=lines[i].replace(' \n','')
        line=line.split(' ')
        if len(line)!=100:
            print(len(line))
            print(line)
            print("Load rel_vec error")
            return
        for j in range(0,100):
            rel_vec[i][j]=float(line[j])
    f=open('./transH/entity2vec.txtbern','r',encoding='utf-8')
    lines=f.readlines()
    f.close()
    if len(lines)!=num_entity:
        print('len(lines)!=entity_num')
    for i in range(0,num_entity):
        line=lines[i].replace(' \n','')
        line=line.split(' ')
        if len(line)!=100:
            print(len(line))
            print(line)
            print("Load rel_vec error")
            return
        for j in range(0,100):
            entity_vec[i][j]=float(line[j])
    #print(A)
    #print(rel_vec)
    print('load_embedding over')
    return A,rel_vec,entity_vec


def load_entity2id():
    f=open('./data/entity2id.txt','r',encoding='utf-8')
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
    print('load_entity2id over')
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

def load_evaluate_data(idx,entity2id,entity_vec,dis2acc):
    tails=[]
    real_acc=[]
    X=[]
    
    all_evaluate_dis=[]
    all_dis=list(set(dis2acc.keys()))
    for dis in dis2acc:
        all_evaluate_dis.append(dis)
        if len(all_evaluate_dis)==500:
            break
    
    dis1_id=all_evaluate_dis[idx]
    head=dis1_id

    real_acc=dis2acc[dis1_id]
        
    for dis2_id in all_dis:
        X.append([entity_vec[dis1_id],entity_vec[dis2_id]])
        tails.append(dis2_id)
    print('load_evaluate_data over')
    return X,np.array(real_acc),head,np.array(tails)


def magnitude(v):#计算向量大小
    return math.sqrt(sum_of_squares(v))

def squared_distance(v,w):#向量平方距离
    return sum_of_squares(vector_subtract(v,w))
def Euclidean_distance(v,w):# 计算两个向量的欧式距离
    return math.sqrt(squared_distance(v,w))

def cos_distance(v,w):# 计算两个向量的余弦距离
    return dot(v,w)/(magnitude(w)*magnitude(v))
def calculate(X,W,r):
    res=[]
    for rel in X:
        head=np.array(rel[0])
        tail=np.array(rel[1])
        h_r=head-np.dot(head,W)*head
        t_r=tail-np.dot(tail,W)*tail#h t在accompany关系平面上的投影
        #并发关系的两个疾病在accompany空间中的位置应该接近，
        #因为TransH训练集中,并发关系是双向的
        distance=abs(h_r+r-t_r).sum()
        res.append(distance)
    return res

num_rel=2
entity2id,num_entity=load_entity2id()
dis2acc=load_dis_acc('./data/train.txt',entity2id)
A,rel_vec,entity_vec=load_embedding(num_rel,num_entity)


MAP=0
P=0
R=0
a=0
k=10

for i in range(0,500):
    print(i)
    X,real_acc,head,tails=load_evaluate_data(i,entity2id,entity_vec,dis2acc)
    #print(np.array(X).shape)
    this_result=np.array(calculate(X,A[0],rel_vec[0]))#判断是并发的概率,这里用欧氏距离
    #print(this_result[0:10])
    this_result=tails[np.argsort(this_result)]#按照误差从低到高排序
    #print(this_result[0:10])
    #print(real_acc)
    MAP+=MAP_1(this_result,real_acc)
    P+=P_k(k,this_result,real_acc)
    R+=R_k(k,this_result,real_acc)
    a+=average_position(this_result,real_acc)
P=P/500
R=R/500
MAP=MAP/500
print("MAP: ",MAP)
print("P_K: ",P)
print("R_K: ",R)
a/=500
print("average_position of real acc: ",a)
'''
distance=abs(h_r+r-t_r).sum():
(500)
MAP:  0.3628279977639344
P_K:  0.1440000000000002
R_K:  0.7022088189588186
average_position of real acc:  13.884614213564213

distance=abs(h_r-t_r).sum()
MAP:  0.04998166499264773
P_K:  0.026000000000000002
R_K:  0.026000000000000002

'''
