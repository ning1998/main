import json
import codecs
import numpy
with codecs.open('./data/dis_dict.json','r',encoding='UTF-8') as f:
    dis_name=json.load(f)
with codecs.open('./data/sym_dict.json','r',encoding='UTF-8') as f:
    dis_name+=json.load(f)
print(len(dis_name))#14805
entity2id={}
entity_num=0
for dis in dis_name:
    if dis not in entity2id:
        entity2id[dis]=entity_num
        entity_num+=1
print(len(entity2id))#14387,有疾病症状名字一样

f=open('./data/entity2id.txt','w+',encoding='utf-8')
for e2id in entity2id:
    f.write(e2id+'\t'+str(entity2id[e2id])+'\n')
f.close()

f=open('./data/relation2id.txt','w+',encoding='utf-8')
f.write('accompany\t0\n'+'has_symptom\t1\n')
f.close()


train_data=[]
with codecs.open('./data/dis_accompany.json','r',encoding='UTF-8') as f:
    rels=json.load(f)
for rel in rels:
    rel=rel.replace('[&&&]','\t')+'\t'+'accompany'
    train_data.append(rel)
with codecs.open('./data/dis_sym.json','r',encoding='UTF-8') as f:
    rels=json.load(f)
for rel in rels:
    rel=rel.replace('[&&&]','\t')+'\t'+'has_symptom'
    train_data.append(rel)
print(len(train_data))#66734
numpy.random.shuffle(train_data)
f=open('./data/train.txt','w+',encoding='utf-8')
for d in train_data:
    f.write(d+'\n')
f.close()

test_data=train_data[0:int(0.2*len(train_data))]
f=open('./data/test.txt','w+',encoding='utf-8')
for d in test_data:
    f.write(d+'\n')
f.close()





