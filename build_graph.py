import re
import os
import json
from py2neo import Graph,Node
import pandas
import codecs
class MedicalGraph:
    def __init__(self):
        self.g = Graph(
            host="localhost",  # neo4j 搭载服务器的ip地址，ifconfig可获取到
            http_port=7474,  # neo4j 服务器监听的端口号
            user="neo4j",  # 数据库user name，如果没有更改过，应该是neo4j
            password="hhn199806029822")
    def create_node(self,desc_path,label):
        print(desc_path)
        dis_desc = []
        file = open(desc_path, 'r', encoding='utf-8')
        for line in file.readlines():
            dic = json.loads(line)
            dis_desc.append(dic)
        print(len(dis_desc))
        my_keys=['name', 'desc', 'category', 'prevent', 'cause',
                 'yibao_status', 'get_prob', 'get_way',
                 'cure_department', 'cure_way', 'cure_lasttime',
                 'cured_prob', 'check', 'recommand_drug']
        count =0
        err=0
        err_name=[]
        for i in range(len(dis_desc)):
            desc=dis_desc[i]
            desc_keys=list(desc.keys())
            desc_values=list(desc.values())
            name=desc['name']
            query="create (n:"+label+" { name: '"+name+"'"
            for j in range(len(desc_keys)):
                key=desc_keys[j]
                if key not in my_keys:
                    continue
                value=desc_values[j]
                S=value
                if type(value)==list:
                    S=""
                    for v in value:
                        S=S+v+"、"
                
                query=query+" ,"+key+": '"+S+"'"
            query=query+"})"
            try:
                self.g.run(query)
                count += 1
            except Exception as e:
                err+=1
                err_name.append(name)
                #print(e)
        print(count)
        print(err_name)
    def create_rel(self,rel_path,start_label,end_label,rel_type, rel_name):
        print(rel_path)
        rels=[]
        with open(rel_path,'r',encoding='UTF-8') as f:
            rels=json.load(f)
        rels2=[]
        for i in rels:
            rel=i.split('[&&&]')
            if len(rel)!=2:
                print("creat rel error "+rel_path)
            rels2.append(rel)
        print(len(rels2))
        count=0
        err=0
        err_name=[]
        for rel in rels2:
            p=rel[0]
            q=rel[1]
            query="match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_label, end_label, p, q, rel_type, rel_name)
            try:
                self.g.run(query)
                count += 1
            except Exception as e:
                err+=1
                err_name.append(rel)
                #print(e)
        print(count)
        print(err_name)
        
    
    def create_node2(self,name_path,label):
        with open(name_path,'r',encoding='UTF-8') as f:
            dis_name=json.load(f)
        print(len(dis_name))
        
        count =0
        err=0
        err_name=[]
        for i in range(len(dis_name)):
            name=dis_name[i]
            query="create (n:"+label+" { name: '"+name+"'"
            
            query=query+"})"
            try:
                self.g.run(query)
                count += 1
            except Exception as e:
                err+=1
                err_name.append(name)
                #print(e)
        print(count)
        print(err_name)
    def create_node3(self,dis_name,label):
        count =0
        err=0
        err_name=[]
        for i in range(len(dis_name)):
            name=dis_name[i]
            query="create (n:"+label+" { name: '"+name+"'"
            
            query=query+"})"
            try:
                self.g.run(query)
                count += 1
            except Exception as e:
                err+=1
                err_name.append(name)
                #print(e)
        print(count)
        print(err_name)
miss_dis=['二尖瓣脱垂综合征', '原发性肝癌', '视网膜色素上皮炎', '唇炎', '心血管疾病', '皮肤白喉', '副牛痘', '不稳定血红蛋白病', '胆道癌', '小儿甲状腺功能亢进症',
 'Hughes-Stovin综合征', '小儿血小板无力症', 'β-氨基酸尿', '乳溢-闭经综合征', '硬化性骨髓炎', '色素性皮肤病', '原发性肾上腺皮质功能减退症', '久疟']

handle=MedicalGraph()
#handle.create_node2('./data/sym_dict.json','症状')
#handle.create_node3(miss_dis,'疾病')
#handle.create_node('./data/medical.json','疾病')
#handle.create_rel('./data/dis_sym.json','疾病','症状','has_symptom','has_symptom')
#handle.create_rel('./data/dis_accompany.json','疾病','疾病','accompany','accompany')
