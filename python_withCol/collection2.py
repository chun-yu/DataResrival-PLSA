import numpy as np
import numba
import gc
import collections
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import non_negative_factorization
from scipy.sparse import issparse, csr_matrix, coo_matrix


def normalize(vec):
    #vec = vec*1.0 / np.linalg.norm(vec) 
    s=np.sum(vec)
    #assert(abs(s) != 0.0) # the sum must not be 0
    for i in range(len(vec)):
        vec[i]=vec[i]*1.0/s

'''
X:sparse matrix of shape(N,V) == dtf矩陣
'''

@numba.njit(fastmath=True,nogil=True)#nogil:當進入這類編譯好的函數時，Numba將會釋放全局線程鎖，fastmath:減少運算時間
def plsa_e_step(X_rows,X_cols,X_vals,p_w_k,p_k_d,p_k_dw,probability_threshold=1e-32):
    K = p_w_k.shape[0] #k:p(Tk|wi,dj)的列，有幾個topic
    for nz in range(X_vals.shape[0]):#對matrix裡非負元素做拜訪
        d=X_rows[nz]#取第d列 foR N
        w=X_cols[nz]#取第w行 for V

        norm=0.0
        for k in range(K):
            temp = p_w_k[k,w]*p_k_d[d,k]
            if temp>probability_threshold:
                p_k_dw[nz,k]=temp
                norm+=temp
            else:
                p_k_dw[nz,k]=0.0
           
        for k in range(K):
            if norm>0:
                p_k_dw[nz,k]/=norm
                
    return p_k_dw

@numba.njit(fastmath=True,nogil=True)
def plsa_m_step(X_rows,X_cols,X_vals,p_w_k,p_k_d,p_k_dw,norm_pwz,norm_pdz):
    K = p_k_dw.shape[1] #k:取 nz x k裡的行
    n = p_k_d.shape[0] #n:全部doc的數量
    m = p_w_k.shape[1] #m:全部word的數量

    p_w_k[:]=0.0
    p_k_d[:]=0.0

    norm_pwz[:]=0.0
    norm_pdz[:]=0.0

    for nz in range(X_vals.shape[0]):
        d=X_rows[nz]
        w=X_cols[nz]
        x=X_vals[nz]

        for k in range(K):
            sum_temp=x*p_k_dw[nz,k]

            p_w_k[k,w]+=sum_temp
            p_k_d[d,k]+=sum_temp

            norm_pdz[d]+=sum_temp
            norm_pwz[k]+=sum_temp

    for k in range(K):
        for w in range(m):
            if norm_pwz[k]>0:
                p_w_k[k,w]/=norm_pwz[k]
        for d in range(n):
            if norm_pdz[d]>0:
                p_k_d[d,k]/=norm_pdz[d]
    
    return p_w_k,p_k_d

@numba.njit(fastmath=True, nogil=True)
def log_likelihood(X_rows, X_cols, X_vals, p_w_k, p_k_d):
    result=0.0
    K=p_w_k.shape[0]
    for nz in range(X_vals.shape[0]):
        d=X_rows[nz]
        w=X_cols[nz]
        x=X_vals[nz]

        p_w_d=0.0
        for k in range(K):
            p_w_d+=p_w_k[k,w]*p_k_d[d,k]
        
        result+=x*np.log(p_w_d)
    return result
@numba.njit(nogil=True)
def plsa_refit_m_step(
    X_rows, X_cols, X_vals, p_w_k, p_k_d, p_k_dw, norm_pdz):

    K = p_k_dw.shape[1]
    n = p_k_d.shape[0]

    p_k_d[:] = 0.0
    norm_pdz[:] = 0.0

    for nz_idx in range(X_vals.shape[0]):
        d = X_rows[nz_idx]
        w = X_cols[nz_idx]
        x = X_vals[nz_idx]

        for z in range(K):
            s = x * p_k_dw[nz_idx, z]
            p_k_d[d, z] += s
            norm_pdz[d] += s

    for z in range(K):
        for d in range(n):
            if norm_pdz[d] > 0:
                p_k_d[d, z] /= norm_pdz[d]

    return p_w_k, p_k_d
if __name__ == "__main__":
    query_txt = open("./query_list.txt")
    doc_txt = open("./doc_list.txt")
    bglm_txt = open("./BGLM.txt")
    collection_txt = open("./Collection.txt")
    query_list = []
    query_temp = []
    doc = lambda: collections.defaultdict(doc)
    doc_dict=doc()
    col = lambda: collections.defaultdict(col)
    collection_dict=col()
    col_w_index_dict={}
    doc_word_index_dict={}
    doc_list = []
    doc_temp = []
    doc_norepeat_word = []
    col_norepeat_word =[]
    col_word_size = []
    query_word =[]
    doc_word_size=[]
    query_length=0
    doc_length=0
    path_q = "./Query"
    path_d = "./Document"

    for line in query_txt.readlines():
        line=line.strip('\n')
        query_list.append(line)
    query_txt.close()

    for line in doc_txt.readlines():
        line=line.strip('\n')
        doc_list.append(line)
    doc_txt.close()

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

    #dtf in collection

    count=0
    c_readlines=collection_txt.readlines()
    doc_i=0
    for line in c_readlines:
        line=line.strip('\n')
        collection_word=line.split()
        collection_temp=list({}.fromkeys(collection_word).keys())
        for c in collection_temp:
            collection_dict["d"+str(doc_i)][c]=collection_word.count(c)
            if(col_w_index_dict.get(c,None)==None):
                col_w_index_dict[c]=count
                count+=1
        doc_i+=1
        col_norepeat_word.append(collection_temp)
        col_word_size.append(collection_word)
    N=len(c_readlines)
    collection_txt.close()
    V=len(col_w_index_dict)
    del c_readlines,doc_i,count
    gc.collect()
    #index_dict = {v: k for k, v in w_index_dict.items()}

    #初始化
    Q=len(query_list)
    K=150
    #N=len(doc_list)
    max_iter=100
    pdj=1.0/N
    #V=len(word_size)
    Pt=np.zeros(K)
    pw_t=np.zeros((K,V),dtype=np.float) #P(w|Tk)
    pd_t=np.zeros((N,K),dtype=np.float) #P(Tk|d)
    #Pt_dw=np.memmap(filename, dtype='float16', mode='w+', shape=(N,V,K))#P(Tk|w,d)
    term_doc_matrix=np.zeros((N,V),dtype=np.int)#dtf
    pwd_t=np.zeros((len(query_list),N),dtype=np.float)
    norm_pwz = np.zeros(K)
    norm_pdz = np.zeros(N)


    pw_t=np.random.random(size=(K,V))
    for j in range(K):
        normalize(pw_t[j])

    pd_t=np.random.random(size=(N,K))
    for j in range(N):
        normalize(pd_t[j])

    for n in range(N):
        for w in range(len(col_norepeat_word[n])):
            w_index = col_w_index_dict.get(col_norepeat_word[n][w],None)
            if(w_index==None):
                print("index=0 error"+col_norepeat_word[n][w]+" w_index:"+str(n))
            count = collection_dict["d"+str(n)].get(col_norepeat_word[n][w],0)
            term_doc_matrix[n][w_index]=count

    X = check_array(term_doc_matrix, accept_sparse="csr")
    if not issparse(X):
        X = csr_matrix(X)
    A = X.tocoo()
    del X
    gc.collect()

    Pt_dw = np.zeros((A.data.shape[0], K))

    for m in range(max_iter):
        print("iteration"+str(m))
        print("E Step:")
        Pt_dw=plsa_e_step(A.row,A.col,A.data,pw_t,pd_t,Pt_dw)
        
        print("M Step:")
    #update P(w|Tk)
        pw_t,pd_t=plsa_m_step(A.row,A.col,A.data,pw_t,pd_t,Pt_dw,norm_pwz,norm_pdz)

        #likelihood=log_likelihood(A.row,A.col,A.data,pw_t,pd_t)
        #print(likelihood)
    del pd_t,Pt_dw,A
    gc.collect()

    count=0
    doc_w_index_dict={}
    #算dtf w_index  把之前的算是放進來
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
            if(doc_w_index_dict.get(z,None)==None):
                doc_w_index_dict[z]=count
                count+=1
        doc_norepeat_word.append(word_1)
        doc_word_size.append(doc_temp)
    
    word_size=list(doc_w_index_dict.keys())
    N=len(doc_list)
    term_doc_matrix=np.zeros((N,V),dtype=np.int)
    for n in range(N):
        for w in range(len(doc_norepeat_word[n])):
            w_index = col_w_index_dict.get(doc_norepeat_word[n][w],None)
            if(w_index==None):
                continue
            else:
                count = doc_dict["d"+str(n)].get(doc_norepeat_word[n][w],0)
                term_doc_matrix[n][w_index]=count
            
    X = check_array(term_doc_matrix, accept_sparse="csr")
    if not issparse(X):
        X = csr_matrix(X)
    A = X.tocoo()

    Pt_dw = np.zeros((A.data.shape[0], K))

    pd_t = np.zeros((A.row.shape[0], K))
    pd_t = np.random.random(size=(N,K))
    for j in range(N):
        normalize(pd_t[j])
    norm_pdz = np.zeros(pd_t.shape[0])
    for m in range(max_iter):
        print("iteration"+str(m))
        print("E Step:")
        Pt_dw=plsa_e_step(A.row,A.col,A.data,pw_t,pd_t,Pt_dw)
        
        print("M Step:")
    #update P(w|Tk)
        pw_t,pd_t=plsa_refit_m_step(A.row,A.col,A.data,pw_t,pd_t,Pt_dw,norm_pdz)
        #likelihood=log_likelihood(A.row,A.col,A.data,pw_t,pd_t)
        #print(likelihood)
    fp=open("plsa_result.txt", "w")
    for q in range(Q):
        for d in range(N):
            for w in range(len(query_word[q])):
                w_index =col_w_index_dict.get(query_word[q][w],None)
                if w_index==None:
                    plsa_prob=0
                else:
                    plsa_prob=pd_t[d,:]*pw_t[:,w_index]
                plsa_prob=np.sum(plsa_prob)
                fp.write(str(q)+" "+str(query_word[q][w])+" "+str(d)+" "+str(plsa_prob)+"\n")
    fp.close() 