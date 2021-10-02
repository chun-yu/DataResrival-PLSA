import os
import re
import numpy as np
import collections
import datetime
np.seterr(divide='ignore', invalid='ignore')
import numba as nb
from numba import jit
starttime = datetime.datetime.now()
query_txt = open("./query_list.txt")
doc_txt = open("./doc_list.txt")
bglm_txt = open("./BGLM.txt")
query_list = []
query_temp = []
query = lambda: collections.defaultdict(query)
query_dict=query()
doc = lambda: collections.defaultdict(doc)
doc_dict=doc() #docTF { docIndex {wordIndex, num } }
Allword_dict={} #All Word index {word , index}
doc_list = []
doc_temp = []
docs_word = []
query_word =[] #query內所有的word [ [23,34,5,...], [], [], [], [], ....,[]]
doc_word_size=[]

#讀檔存資料
for line in query_txt.readlines():
    line=line.strip('\n')
    query_list.append(line)
query_txt.close()

for line in doc_txt.readlines():
    line=line.strip('\n')
    doc_list.append(line)
doc_txt.close()


count=0
word_size=[]

#算qtf 
for i in range(len(query_list)):
    query_temp=[]
    f = open("./Query/"+query_list[i])
    for line in f.readlines():
        line=line.strip('\n')
        query_temp.append(line.split())
    f.close()
    query_temp=[y for x in query_temp for y in x]
    query_temp=[z for z in query_temp if z!='-1']
    query_word.append(query_temp)

#算dtf idf
for i in range(len(doc_list)):
    d = open("./Document/"+doc_list[i],'r')
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
        if(Allword_dict.get(z,None)==None):
            Allword_dict[z]=count
            count+=1
    docs_word.append(word_1)   #doc word 排除重複
    doc_word_size.append(doc_temp)  #doc word 不排除重複
word_size=list(Allword_dict.keys())
print(count)
print(len(word_size))
#初始化=========================
Q=len(query_list)
K=50
D=len(doc_list)
max_iter=100 #疊代次數
a=0.1
b=0.395
W=len(word_size)  #wi 總數 13290
Pt=np.zeros(K)
pw_t=np.zeros((K,W),dtype=np.float) #P(w|Tk)
pt_d=np.zeros((D,K),dtype=np.float) #P(Tk|d)
Pt_w_d=np.zeros((D,W,K),dtype=np.float)#P(Tk|w,d)
term_doc_matrix=np.zeros((D,len(word_size)),dtype=np.int)#dtf
pwd_t=np.zeros((len(query_list),D),dtype=np.float)

pw_t=np.random.random(size=(K,W))
for j in range(K):
    pw_t[j]=pw_t[j] / (np.sum(pw_t[j]))
    
pt_d=np.random.random(size=(D,K))
for j in range(D):
    pt_d[j]=pt_d[j] / (np.sum(pt_d[j]))
    
#dtf 
for n in range(D):
    for w in range(len(docs_word[n])):
        w_index = Allword_dict.get(docs_word[n][w],None)
        if(w_index==None):
            print("index=0 error"+docs_word[n][w]+"w_index:"+str(n))
        count = doc_dict["d"+str(n)].get(docs_word[n][w],0)
        term_doc_matrix[n][w_index]=count           
#run EM
# E Step========
for m in range(max_iter):
    print("EM  "+str(m))
    #print("E Step:")
    for d_index in range(D):
        prob=np.transpose([pt_d[d_index]])
        upl=np.multiply(pw_t,prob)#upl=pw_t*prob
        upl=np.transpose(upl)
        nor=upl.sum(axis=1)
        nor=np.transpose([nor])
        upl=np.true_divide(upl, nor) #normalize
        Pt_w_d[d_index]=upl
        del upl,nor,prob
    #print("M Step:")
# M Step========
#P(w|Tk)
    for k in range(K):
        count=Pt_w_d[:,:,k]
        s=np.multiply(term_doc_matrix,count)
        s= s.sum(axis=0)
        pw_t[k]=s
        pw_t[k]=np.true_divide(pw_t[k], (np.sum(pw_t[k]))) #normalize
        del s,count
#P(Tk|d)
    for d_index in range(D):
        count=term_doc_matrix[d_index]
        count=np.transpose([count])
        s=np.multiply(Pt_w_d[d_index],count)#s=Pt_w_d[d_index]*count
        s= s.sum(axis=0)
        pt_d[d_index]=np.true_divide(s, (count.sum(axis=0))) #normalize
        del s,count
#完成 EM
print("EM finish !!")

#儲存BGLM 值
BGLM={}
for line in bglm_txt.readlines():
    (key, val) = line.split()
    BGLM[int(key)] = val
fp = open("submission.txt", "w")
fp.write("Query,RetrievedDocuments")
print ("write submission")
#寫出判斷排序檔
for q in range(Q):
    print ("qurey %d"%(q))
    for d in range(D):
        s=0
        for w in range(len(query_word[q])):
            w_index = Allword_dict.get(query_word[q][w],0)
            if(w_index==0):
                plsa_prob=0
            else:
                plsa_prob=np.multiply(pt_d[d,:],pw_t[:,w_index])#plsa_prob=pt_d[d,:]*pw_t[:,w_index]
                plsa_prob=np.sum(plsa_prob)
            pwd=doc_dict["d"+str(d)].get(query_word[q][w],0)/len(doc_word_size[d])
            pbglm=BGLM.get(int(query_word[q][w]),0)
            s1=(np.log(a) + np.log(pwd ))
            s2=(np.log(b) + np.log(plsa_prob ))
            s3=(np.log(1-a-b) + float(pbglm))
            s= (np.logaddexp(np.logaddexp(s1,s2),s3)) + s
        pwd_t[q][d]=s
    sort_index=np.argsort(pwd_t[q])
    sort_index=sort_index.tolist()
    sort_index.reverse()
    fp.write("\n"+query_list[q]+",")
    for k in range(len(doc_list)):
        fp.write(doc_list[sort_index[k]]+" ")
fp.close()
print ("finish write")
endtime = datetime.datetime.now()
print ((endtime-starttime).seconds)