import math
import numpy as np
from keras.preprocessing.sequence import pad_sequences


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

def load_train_test(entity2id,id2embed):
    f=open('../train_data/train_data.txt','r',encoding='utf-8')
    data=f.readlines()
    f.close()
    X=[]
    y=[]
    step=0
    for d in data:
        if step%1000==0:
            print(step)
        step+=1
        this_X=[]
        S=d.replace('\n','').split('\t')
        if len(S)!=3:
            print('error1',S)
        this_y=int(S[2])
        dis1_id=entity2id[S[0]]
        dis2_id=entity2id[S[1]]
        if dis2_id not in id2embed:#train_data.txt中负采样的疾病结点可能原本是孤立点，network的embedding中没有它
            continue
        this_X=[id2embed[dis1_id],id2embed[dis2_id]]
        X.append(this_X)
        y.append(this_y)
    train_X=np.array(X[0:int(0.8*len(X))])
    train_y=y[0:int(0.8*len(y))]
    test_X=np.array(X[int(0.8*len(X)):])
    test_y=y[int(0.8*len(X)):]
    return train_X,train_y,test_X,test_y

        
num_rel=2
entity2id,num_entity=load_entity2id()

id2embed=load_embedding()
train_X,train_y,test_X,test_y=load_train_test(entity2id,id2embed)
#print(train_X[0])
print(train_X.shape,test_X.shape)#(?,2,100)


from keras.optimizers import Adagrad, Adam, SGD, RMSprop, Nadam
from keras.callbacks import EarlyStopping
from my_model import predict_model

print('Build model...')
model = predict_model(embedding_dim=100,class_num=2,dis_embedding=128).get_model()
model.load_weights('./model/my_model')
model.compile(optimizer = SGD(lr = 0.001, decay = 1e-4),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
best_p=0#0.9692
result = model.predict(test_X)
result=result.argmax(axis=1)
best_p=np.sum(result==test_y)/result.shape[0]

print("初始 : accuracy=",best_p)
batch_size=128
epochs=10
print('Train...')
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
for i in range(0,epochs):
    model.fit(train_X, train_y,
              batch_size=batch_size,
              epochs=1,
              callbacks=[early_stopping],
              validation_data=(test_X, test_y))
    print('Test...')
    result = model.predict(test_X)
    result=result.argmax(axis=1)
    p=np.sum(result==test_y)/result.shape[0]
    print("epoch "+str(i)+": accuracy=",p)
    if p>best_p:
        best_p=p
        model.save_weights('./model/my_model')

















