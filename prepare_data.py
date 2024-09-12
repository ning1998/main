import json
import codecs
import numpy

#prepare  data
#entity2label  entity label1 label2
#edgelist head tail rel_label

entity2label={}
adjlist={}
with codecs.open('../data/dis_dict.json','r',encoding='UTF-8') as f:
    dis_name=json.load(f)
with codecs.open('../data/sym_dict.json','r',encoding='UTF-8') as f:
    sym_name=json.load(f)
f=open('../data/entity2id.txt','r',encoding='utf-8')
lines=f.readlines()
f.close()
entity2id={}
for line in lines:
    word=line.replace('\n','').split('\t')
    entity2id[word[0]]=int(word[1])
print(len(list(set(dis_name+sym_name))))#14387
print(len(list(entity2id.keys())))#14387
for word in entity2id:
    word_id=entity2id[word]
    entity2label[word_id]=[]
    if word in dis_name:
        entity2label[word_id].append('disease')
    if word in sym_name:
        entity2label[word_id].append('symptom')
f=open('./data/entity2label.txt','w+',encoding='utf-8')
for word_id in entity2label:
    S=str(word_id)+' '+' '.join(entity2label[word_id])+'\n'
    f.write(S)
f.close()


edgelist=[]
with codecs.open('../data/dis_accompany.json','r',encoding='UTF-8') as f:
    rels=json.load(f)
rels=list(set(rels))
for rel in rels:
    head,tail=rel.split('[&&&]')
    head=entity2id[head]
    tail=entity2id[tail]
    edgelist.append(str(head)+' '+str(tail)+' '+'dis_accompany'+'\n')

with codecs.open('../data/dis_sym.json','r',encoding='UTF-8') as f:
    rels=json.load(f)
rels=list(set(rels))
for rel in rels:
    head,tail=rel.split('[&&&]')
    head=entity2id[head]
    tail=entity2id[tail]
    edgelist.append(str(head)+' '+str(tail)+' '+'dis_sym'+'\n')
f=open('./data/edgelist.txt','w+',encoding='utf-8')
for edge in edgelist:
    f.write(edge)
f.close()
print(len(edgelist))#66734
