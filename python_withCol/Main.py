# !/usr/bin/python
# coding:utf-8
import os
import re
import numpy as np
import math
import collections
import datetime
import pandas as pd
from multiprocessing import Pool,cpu_count
starttime = datetime.datetime.now()
#query_txt = open("C:/Users/xuan9/Homework3/query_list.txt")
query_txt = open("./query_list.txt")
#doc_txt = open("C:/Users/xuan9/Homework3/doc_list.txt")
doc_txt = open("./doc_list.txt")
bglm_txt = open("./BGLM.txt")
plsa_txt = open("./plsa_result.txt")
path_q = "./Query"
path_d = "./Document"

query_list=[]
doc_list=[]
query_word=[]
query = lambda: collections.defaultdict(query)
query_dict=query()
doc = lambda: collections.defaultdict(doc)
doc_dict=doc()
word_index_dict={}
docs_word_Norepeat=[]
doc_word_size=[]

for line in query_txt.readlines():
    line=line.strip('\n')
    query_list.append(line)
query_txt.close()

for line in doc_txt.readlines():
    line=line.strip('\n')
    doc_list.append(line)
doc_txt.close()




BGLM={}
for line in bglm_txt.readlines():
    (key, val) = line.split()
    BGLM[int(key)] = float(val)

#算qtf 
for i in range(len(query_list)):
    query_temp=[]
    f = open(path_q+"/"+query_list[i])
    for line in f.readlines():
        line=line.strip('\n')
        query_temp.append(line.split())
    f.close()
    query_temp=[y for x in query_temp for y in x]
    query_temp=[z for z in query_temp if z!='-1']
    query_word.append(query_temp)
    count=0
    word_size=[]

#算dtf idf
for i in range(len(doc_list)):
    d = open(path_d+"/"+doc_list[i],'r')
    doc_temp=[]
    dict_temp={}
    for line in d.readlines():
        line=line.strip('\n')
        doc_temp.append(line.split())
    d.close()
    doc_temp=[y for x in doc_temp for y in x]
    doc_temp=[z for z in doc_temp if z!='-1']
    doc_temp=doc_temp[5:]
    dict_temp=dict_temp.fromkeys(doc_temp)
    word_1=list(dict_temp.keys())
    for z_index,z in enumerate(word_1):
        doc_dict["d"+str(i)][z]=doc_temp.count(z)
        if(word_index_dict.get(z,None)==None):
            word_index_dict[z]=count
            count+=1
    docs_word_Norepeat.append(word_1)
    doc_word_size.append(doc_temp)
word_size=list(word_index_dict.keys())

Q=len(query_list)
N=len(doc_list)
b=0.325
a=0.105

V=len(word_size)
pwd_t=np.zeros((len(query_list),N),dtype=np.float)

plsa_dict={}
plsa_dict=plsa_dict.fromkeys(list(range(0,Q)),{})
for q_index in range(Q):
    for n_index in range(N):
        plsa_dict[q_index][str(n_index)]={}

for line in plsa_txt.readlines():
    (q_number, q_word , doc_number , plsa_value ) = line.split()
    plsa_dict[int(q_number)][str(doc_number)][q_word]=float(plsa_value)
 
#計算p(wi|dj)
fp = open("submission.txt", "w")
fp.write("Query,RetrievedDocuments")
for q in range(Q):
    for d in range(N):
        s=0
        for w in range(len(query_word[q])):
            w_index = word_index_dict.get(query_word[q][w],None)
            if(w_index==None):
                plsa_prob=0
            else:
                plsa_prob= plsa_dict[q][str(d)].get(query_word[q][w],None)
                if(plsa_prob==None):
                    print("error")
            pwd=float(doc_dict["d"+str(d)].get(query_word[q][w],0))/len(doc_word_size[d])
            pbglm=BGLM.get(int(query_word[q][w]),0)
            '''
            if(pwd>0 & plsa_prob>0):
                s1=np.log(a)+np.log(pwd)
                s2=np.log(b)+np.log(plsa_prob)
                s1=np.logaddexp(s1,s2)
            else:
                s1=0
            if(plsa_prob>0):
                
            else:
                s2=0
            '''
            s1=np.log(a)+np.log(pwd)
            s2=np.log(b)+np.log(plsa_prob)
            s1=np.logaddexp(s1,s2)
            s2=np.log(1-a-b)+pbglm
            s1=np.logaddexp(s1,s2)
            s=s+s1          
        pwd_t[q][d]=s
    sort_index=np.argsort(pwd_t[q])
    sort_index=sort_index.tolist()
    sort_index.reverse()
    fp.write("\n"+query_list[q]+",")
    for k in range(len(doc_list)):
        fp.write(doc_list[sort_index[k]]+" ")
fp.close()