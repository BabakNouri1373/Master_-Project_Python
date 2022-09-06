#!/usr/bin/env python
# coding: utf-8

# In[1]:


#------------------------------------------------------
#             Start The Project             
#------------------------------------------------------


# In[2]:


#------------------------------------------------------
#             Import Required Libary                  
#------------------------------------------------------
import math
import numpy  as np
import pandas as pd
import time   as ti
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import networkx as nx
from sklearn.neighbors import NearestNeighbors
#------------------------------------------------------


# In[3]:


#------------------------------------------------------
#             Define For Select Row Of Matrix                    
#------------------------------------------------------
def Index_Select(matrix, i):
    return [row[i] for row in matrix]
#------------------------------------------------------


# In[4]:


#------------------------------------------------------
#             Calculate Count Of Cluster          
#------------------------------------------------------
def Cluster_Count(Main_Labels):
    
    input_list=Main_Labels
    l1 = []
    count = 0
    for item in input_list:
        if item not in l1:
            count += 1
            l1.append(item)
    Cluster_Count=count
    return Cluster_Count
#------------------------------------------------------


# In[5]:


#------------------------------------------------------
#             Definishion K_Means Algoritm (Clustering)                 
#------------------------------------------------------
def K_Means_Clustering(Data_Set,Count_Of_Cluster):
    
    kmeans = KMeans(Count_Of_Cluster)
    kmeans.fit(Data_Set)
    identified_clusters = kmeans.fit_predict(Data_Set)
    kmeans=pd.DataFrame(identified_clusters, index= None)
    kmeans_Label=np.matrix(kmeans)
    return kmeans_Label
#------------------------------------------------------


# In[6]:


#------------------------------------------------------
#             Definishion accuracy                 
#------------------------------------------------------
def Acc(Main_Labels,K_Labels):
    
    P=Main_Labels
    Q=np.array(K_Labels).ravel().tolist()
    
    return accuracy_score(P,Q)
#------------------------------------------------------


# In[7]:


#------------------------------------------------------
#             Definishion normalized mutualinformation              
#------------------------------------------------------
def NMI(Main_Labels,K_Labels):
    
    P=Main_Labels
    Q=np.array(K_Labels).ravel().tolist()
    I_PQ=mutual_info_score(P,Q)
    H_P=entropy(P)
    H_Q=entropy(Q)
    
    return I_PQ/((H_P*H_Q)**(1/2))
#------------------------------------------------------


# In[8]:


#------------------------------------------------------
#             Definision D in MFFS Article
#------------------------------------------------------
def D_Value(K,W):
    Matrix=np.matrix(W)
    S=np.dot(W.T,W)
    S_List=[]
    for i in range(K):
        S_List.append(S[i][i])
    diag=np.diag(S_List)
    D=(diag**(1/2))
    return D
#------------------------------------------------------


# In[9]:


#------------------------------------------------------
#             Definision MFFS Algoritm                  
#------------------------------------------------------
def MFFS(X,K,Steps,Alpha,Break_Value):

    #-----------
    #Inter Parametr And Set  Need Value
    #-----------
    X                   =  np.array(Main_Data_Set)
    Y                   =  np.array(Main_Data_Set)
    Row_Number          =  len(X)
    Column_Number       =  len(X[0])
    W                   =  np.random.rand(Column_Number,K)
    H                   =  np.random.rand(K,Column_Number)
    W_Final_Norm        =  []
    W_Final_Norm_Index  =  []
    F_List              =  []
    F_Steps_List        =  []
    #-----------
    #End
    #-----------
    
    
    #-----------  
    #Start Algoritm  For Update
    #-----------
    Start=ti.time()
    for i in range(Steps):
        
        
        #----------- 
        #  Set D
        #----------- 
        D=D_Value(K,W)
        #----------- 
        
        
        #----------- 
        #  Set W
        #----------- 
        W=np.dot(W,np.linalg.inv(D))
        #----------- 
        
        #----------- 
        #  Set H
        #----------- 
        H=np.dot(D,H)
        #----------- 

        
        #----------- 
        #  Update W
        #----------- 
        W_UP=np.dot(np.dot(X.T,X),H.T) + Alpha*W
        W_DOWN=np.dot(np.dot(np.dot(np.dot(X.T,X),W),H),H.T) + Alpha*(np.dot(np.dot(W,W.T),W))
        W=W*(W_UP/W_DOWN)
        #----------- 
        
        
        #----------- 
        #  Update H
        #----------- 
        H_UP=(np.dot(np.dot(W.T,X.T),X))
        H_DOWN=np.dot(np.dot(np.dot(np.dot(W.T,X.T),X),W),H)
        H=H*(H_UP/H_DOWN)
        #----------- 
        
        
        #----------- 
        # Calculate Error(E=X-XWH)
        #----------- 
        Error=np.linalg.norm(X-(np.dot(np.dot(X,W),H)))
        #----------- 
        
        
        #----------- 
        #  Calculate F Function in MFFS Article
        #-----------
        F_Part_1= (1/2)*((np.linalg.norm(X-np.dot(np.dot(X,W),H)))**2)
        F_Part_2=(Alpha/4)*((np.linalg.norm(np.dot(W,W.T)-Error))**2)
        F=F_Part_1+F_Part_2
        F_List.append(F)
        F_Steps_List.append(i)
        #-----------
        
        
        #-----------
        # Define If For Break
        #-----------
        if Error<=Break_Value:
            break
        #-----------
    #-----------
    #End Algoritm  For Update
    #-----------  
    

    
    
    
    
    
    
    #-----------
    #  Calculate Norm OF W And Sort Index And Norm
    #-----------
    for i in range(0,Column_Number):
        
        #-----------
        # Append Norm And Index To List
        #-----------  
        W_Norm          = np.linalg.norm(W[i])
        W_Final_Norm.append(W_Norm)
        W_Final_Norm_Index.append(i+1)
        
    #-----------
    #  Sort W List By Norm Value
    #-----------      
    W_Final_Norm         = np.array(W_Final_Norm)
    W_Final_Norm_Index   = np.array(W_Final_Norm_Index)
    W_Norm_Index         = np.array([[W_Final_Norm],[W_Final_Norm_Index]])
    W_Norm_Index         = W_Norm_Index.T
    W_Norm_Index         = np.matrix(W_Norm_Index)
    W_Sorted             = W_Norm_Index[np.argsort(W_Norm_Index.A[:, 0])]
    W_Sorted             = np.array(W_Sorted)
    Final_Index          = Index_Select(W_Sorted,1)
    Final_Index.reverse()
    W_Sorted             = np.array(W_Sorted)
    Final_Norm           = Index_Select(W_Sorted,0)
    Final_Norm.reverse()
    #-----------  
    #End
    #----------- 
    
    
    
    
    
    
    
    #-----------  
    #Select Need Dimension Of Main Dataset  And Show    
    #-----------
    Dimension_Index_List =[]
    Final_Index_List     =[]
    for i in range(K):
        My_Index=int(Final_Index[i]-1)
        Dimension_Index_List.append(My_Index)
        Final_Index_List.append(int(Final_Index[i]))
    Data_Set_Main    = np.array(Y.T)
    Selected_Column  = Data_Set_Main[Dimension_Index_List]
    return Selected_Column.T
    #-----------  
    #End  
    #-----------
    
    
#------------------------------------------------------


# In[10]:


#------------------------------------------------------
#     Definision For Return MFFS Dataset With Multi K                 
#------------------------------------------------------
def Multi_MFFS(K):
    Data_Set=MFFS(Main_Data_Set,K,Steps,Alpha,Break_Value)
    return  Data_Set
#------------------------------------------------------


# In[11]:


#------------------------------------------------------
#     Definision For Return Multi MFFS             
#------------------------------------------------------
def Data_Sets_MFFS(K):
    return Multi_MFFS(K)
#------------------------------------------------------


# In[12]:


#------------------------------------------------------
#             Definision KNN Algorim                  
#------------------------------------------------------
def KNN(Matrix_Graph,K,Kind):
    knn = NearestNeighbors(algorithm='auto',  n_neighbors=K, p=2);
    knn.fit(Matrix_Graph);
    distances, indices = knn.kneighbors(Matrix_Graph);
    if Kind==0:
        return distances
    if Kind==1:
        return  indices
#------------------------------------------------------


# In[13]:


#------------------------------------------------------
#             Definision W In SLSDR Article                  
#------------------------------------------------------
def W(X,K,sigma):
    n=len(X);
    K=5
    W = np.zeros((n, n));
    distances =KNN(X,K,0)
    indices   =KNN(X,K,1)
    for i in range(n):
        b=indices[i];
        for j in range(K):
            W[i,b[j]] = math.exp(-((distances[i,j])**2)/(sigma**2));
            W[b[j],i] = math.exp(-((distances[i,j])**2)/(sigma**2));
    return W
#------------------------------------------------------


# In[14]:


#------------------------------------------------------
#             Definision For  U_Value Function               
#------------------------------------------------------
def Chech_Zero(Number):
    return max(Number,10**(-8))
#------------------------------------------------------


# In[15]:


#------------------------------------------------------
#             Calculate Diagonal Of Matrix              
#------------------------------------------------------
def Diagonal_Matrix(First_Matrix):
    D_Norm_List=[]
    for i in range(0,len(First_Matrix)):
                   D_Norm= np.linalg.norm(First_Matrix[i])
                   D_Norm_List.append(D_Norm)
    return np.diag(D_Norm_List)
#------------------------------------------------------


# In[16]:


#------------------------------------------------------
#             Definision U in SLSDR Article                  
#------------------------------------------------------
def U_Value(X,V,S):
    E=X-(np.dot(X,np.dot(S,V)))
    E_Norm_List=[]
    for i in range(0,len(E)):
        U_Norm          = np.linalg.norm(E[i])
        U_ii=1/ Chech_Zero(U_Norm)
        E_Norm_List.append(U_ii)
    return np.diag(E_Norm_List)
#------------------------------------------------------


# In[17]:


#------------------------------------------------------
#          Definision SLSDR Algoritm               
#------------------------------------------------------
def SLSDR(X,K,Steps,Alpha,Beta,Lambda,Break_Value,sigma):

    #-----------
    #Inter Parametr And Set  Need Value
    #-----------
    X                   =  np.array(X)
    Y                   =  np.array(X)
    Row_Number          =  len(X)
    Column_Number       =  len(X[0])
    S                   =  np.random.rand(Column_Number,K)
    V                   =  np.random.rand(K,Column_Number)
    S_Final_Norm        =  []
    S_Final_Norm_Index  =  []
    #-----------
    #End
    #-----------
    
    
    #-----------  
    #Start Algoritm
    #-----------
    Start=ti.time()
    for i in range(Steps):
        
        
        #-----------
        #  Set W_S
        #-----------
        W_S=W(X,K,sigma)
        #-----------
        
        
        #-----------
        #  Set D_S
        #-----------
        D_S=Diagonal_Matrix(W_S)
        #-----------
        
        
        #-----------
        #  Set W_V
        #-----------
        W_V=W(X.T,K,sigma)
        #-----------
        
        
        #-----------
        #  Set D_V
        #-----------
        D_V=Diagonal_Matrix(W_V)
        #-----------
    
    
        #-----------
        #  Set U
        #-----------
        U=U_Value(X,V,S)
        #----------- 
        
        
        
        
        #-----------
        #  Update S
        #----------- 
        S_UP1     = np.dot(np.dot(np.dot(X.T,U),X),V.T)
        S_UP2     = np.dot(Alpha * (np.dot(np.dot(X.T,W_S),X)) + ((np.identity(Column_Number)) *  (Beta + Lambda)   ),S) 
        S_UP      = S_UP1 + S_UP2
        
        
        S_DOWN1   = np.dot(np.dot(np.dot(np.dot(np.dot(X.T,U),X),S),V),V.T) 
        S_DOWN2   = np.dot((Alpha*(np.dot(np.dot(X.T,D_S),X))) + (Beta*np.ones(Column_Number))+ Lambda *np.dot(S,S.T),S)
        S_DOWN    = S_DOWN1+S_DOWN2
        
        S         =S*(S_UP/S_DOWN)
        #----------- 
 
        
    
    
    
        #----------- 
        #  Update V
        #----------- 
        V_UP      =np.dot(np.dot(np.dot(S.T,X.T),U),X) +   Alpha*(np.dot(V,W_V))
        
        V_DOWN    =np.dot(np.dot(np.dot(np.dot(np.dot(S.T,X.T),U),X),S),V)  +  Alpha*(np.dot(V,D_V))
                                                                                  
        V         =V*(V_UP/V_DOWN)   
        #----------- 
        
        
        
        
        
        
    #-----------
    #Calculate Norm OF W And Sort Index And Norm
    #-----------
    for i in range(0,Column_Number):
        S_Norm          = np.linalg.norm(S[i])
        S_Final_Norm.append(S_Norm)
        S_Final_Norm_Index.append(i+1)
        
        
    S_Final_Norm         = np.array(S_Final_Norm)
    S_Final_Norm_Index   = np.array(S_Final_Norm_Index)
    S_Norm_Index         = np.array([[S_Final_Norm],[S_Final_Norm_Index]])
    S_Norm_Index         = S_Norm_Index.T
    S_Norm_Index         = np.matrix(S_Norm_Index)
    S_Sorted             = S_Norm_Index[np.argsort(S_Norm_Index.A[:, 0])]
    S_Sorted             = np.array(S_Sorted)
    Final_Index          = Index_Select(S_Sorted,1)
    Final_Index.reverse()
    S_Sorted             = np.array(S_Sorted)
    Final_Norm           = Index_Select(S_Sorted,0)
    Final_Norm.reverse()
    #-----------  
    #End
    #----------- 
    
    
    
    #-----------  
    # Select Need Dimension Of Main Dataset  And  Show    
    #-----------
    Dimension_Index_List=[]
    Final_Index_List=[]
    for i in range(K):
        My_Index=int(Final_Index[i]-1)
        Dimension_Index_List.append(My_Index)
        Final_Index_List.append(int(Final_Index[i]))
    Data_Set_Main=np.array(Y.T)
    Selected_Column=Data_Set_Main[Dimension_Index_List]
    return Selected_Column.T
    #-----------  
    #End  
    #-----------
    
    
#------------------------------------------------------


# In[18]:


#------------------------------------------------------
#           Definision For Return SLSDR Dataset With Multi K              
#------------------------------------------------------
def Multi_SLSDR(K):
    Data_Set=SLSDR(Main_Data_Set,K,Steps,Alpha,Beta,Lambda,Break_Value,sigma)
    return  Data_Set
#------------------------------------------------------


# In[19]:


#------------------------------------------------------
#           Definision For Return Multi SLSDR           
#------------------------------------------------------
def Data_Sets_SLSDR(K):
    return Multi_SLSDR(K)
#------------------------------------------------------


# In[20]:


#------------------------------------------------------
#          Definision  For Return Final Result                 
#------------------------------------------------------
def Result(Main_Labels,Dimension_Select_List,Kmeans_Count,sigma):
    
    
    
    
    #-----------  
    #   Define Empty List For BaseLine Method
    #----------- 
    ACC_List_BaseLine     =[]
    NMI_List_BaseLine     =[]
    ACC_Std_BaseLine      =[]
    NMI_Std_BaseLine      =[]
    #----------- 
    
    
    
    #-----------  
    #   Define Empty List For MFFS Method
    #----------- 
    Multi_ACC_List_MFFS   =[]
    Multi_NMI_List_MFFS   =[]
    ACC_List_MFFS         =[]
    NMI_List_MFFS         =[]
    ACC_Std_MFFS          =[]
    NMI_Std_MFFS          =[]
    #----------- 
    
    
    
    #-----------  
    #   Define Empty List For SLSDR Method
    #----------- 
    Multi_ACC_List_SLSDR  =[]
    Multi_NMI_List_SLSDR  =[]
    ACC_List_SLSDR        =[]
    NMI_List_SLSDR        =[]
    ACC_Std_SLSDR         =[]
    NMI_Std_SLSDR         =[]
    #----------- 
    
    
    
    
    
    #-----------  
    #   Calculate  Acc And NMI For BaseLine Method
    #----------- 
    K_Means_BaseLine =K_Means_Clustering(Main_Data_Set,Cluster_Count(Main_Labels))
    K_Labels_BaseLine=K_Means_BaseLine
    for ij in range(len(Dimension_Select_List)):
        
        ACC_List_BaseLine.append(Acc(Main_Labels,K_Labels_BaseLine))
        NMI_List_BaseLine.append(NMI(Main_Labels,K_Labels_BaseLine))
    #----------- 
    
    
    
    
    
    
    #-----------  
    #   Calculate  Acc And NMI For MFFS  And SLSDR  Method
    #-----------
    
    #-----------  
    #  Define i(Dimension_Select_List.  For Example:[1,12]. Select 1 AND 12 Important Dimention .)
    #----------- 
    for i in Dimension_Select_List:
        
        
        #-----------  
        #  Set Dataset
        #----------- 
        Data_Set_MFFS  =Data_Sets_MFFS(i)
        Data_Set_SLSDR =Data_Sets_SLSDR(i)
        #----------- 
        
        
        
        #-----------  
        # Define j(Kmeans_Count. For Example:(5).  It Sends Each Dataset To The Kmeans Algoritm  5 Times )
        #----------- 
        for j in range(Kmeans_Count):
            
            
            
            #-----------  
            #  Send MFFS To Acc And NMI 
            #----------- 
            K_Means_MFFS =K_Means_Clustering(Data_Set_MFFS,Cluster_Count(Main_Labels))
            K_Labels_MFFS=K_Means_MFFS
            Multi_ACC_List_MFFS.append(Acc(Main_Labels,K_Labels_MFFS))
            Multi_NMI_List_MFFS.append(NMI(Main_Labels,K_Labels_MFFS))
            
            
            
            
            
            #-----------  
            #   Send SLSDR To Acc And NMI 
            #----------- 
            K_Means_SLSDR =K_Means_Clustering(Data_Set_SLSDR,Cluster_Count(Main_Labels))
            K_Labels_SLSDR=K_Means_SLSDR
            Multi_ACC_List_SLSDR.append(Acc(Main_Labels,K_Labels_SLSDR))
            Multi_NMI_List_SLSDR.append(NMI(Main_Labels,K_Labels_SLSDR))
            
            
            
            
            
        #-----------  
        #   Calculate Mean Of MFFS Dateset(Acc And NMI) And Clear List
        #-----------     
        ACC_List_MFFS.append(np.mean(Multi_ACC_List_MFFS))
        NMI_List_MFFS.append(np.mean(Multi_NMI_List_MFFS))
        Multi_ACC_List_MFFS.clear()
        Multi_NMI_List_MFFS.clear()
        #-----------   
       
        
        
        
        
        #-----------  
        #    Calculate Mean Of SLSDR Dateset(Acc And NMI) And Clear List
        #-----------     
        ACC_List_SLSDR.append(np.mean(Multi_ACC_List_SLSDR))
        NMI_List_SLSDR.append(np.mean(Multi_NMI_List_SLSDR))
        Multi_ACC_List_SLSDR.clear()
        Multi_NMI_List_SLSDR.clear()
        #-----------  
  


    #----------- 
    #   Calculate STD OF ACC And NMI For BaseLine And MFFS And SLSDR 
    #-----------
    ACC_Std_BaseLine.append(np.std(ACC_List_BaseLine))     
    NMI_Std_BaseLine.append(np.std(NMI_List_BaseLine))         
    ACC_Std_MFFS.append(np.std(ACC_List_MFFS))
    NMI_Std_MFFS.append(np.std(NMI_List_MFFS))
    ACC_Std_SLSDR.append(np.std(ACC_List_SLSDR))
    NMI_Std_SLSDR.append(np.std(NMI_List_SLSDR))
    #----------- 
    
    
    #----------- 
    #Show ACC in 2*D
    #-----------
    X_List_Acc=Dimension_Select_List
    Y1_List_Acc=ACC_List_BaseLine
    Y2_List_Acc=ACC_List_MFFS
    Y3_List_Acc=ACC_List_SLSDR
    plt.plot(X_List_Acc,Y1_List_Acc,color='lightcoral',marker='D',markeredgecolor='red' , label='BaseLine')
    plt.plot(X_List_Acc,Y2_List_Acc,color='lightcoral',marker='D',markeredgecolor='blue'  , label='MFFS')
    plt.plot(X_List_Acc,Y3_List_Acc,color='lightcoral',marker='D',markeredgecolor='green'  , label='SLSDR')
    plt.ylim(0,0.2)
    plt.xlim(20,100) 
    plt.xlabel('Count Of Dimension') 
    plt.ylabel('Acc Value') 
    plt.title('ACC') 
    plt.legend()
    plt.show()
    #-----------  
    #End
    #-----------  
    
    
    
    #----------- 
    #Show Show NMI in 2*D
    #-----------
    X_List_NMI=Dimension_Select_List
    Y1_List_NMI=NMI_List_BaseLine
    Y2_List_NMI=NMI_List_MFFS
    Y3_List_NMI=NMI_List_SLSDR
    plt.plot(X_List_NMI,Y1_List_NMI,color='lightcoral',marker='D',markeredgecolor='red' , label='BaseLine')
    plt.plot(X_List_NMI,Y2_List_NMI,color='lightcoral',marker='D',markeredgecolor='blue', label='MFFS')
    plt.plot(X_List_NMI,Y3_List_NMI,color='lightcoral',marker='D',markeredgecolor='green' , label='SLSDR')
    plt.ylim(0.2,0.6) 
    plt.xlim(20,100) 
    plt.xlabel('Count Of Dimension') 
    plt.ylabel('NMI Value') 
    plt.title('NMI') 
    plt.legend()
    plt.show()
    #-----------  
    #End
    #-----------
#------------------------------------------------------


# In[21]:


#------------------------------------------------------
#           Enter First Data And Run Algoritm            
#------------------------------------------------------
#   Main_Data_Set = [
#                   [5,3,0,1],
#                   [4,0,0,1],
#                   [1,1,0,5],
#                   [1,0,0,4],
#                  ]


#Main_Labels =[1,0,0,2]




#-----------  
#  Set Main Dataset
#-----------
Main_Data_Set =pd.read_csv('C:\\Users\\babak_Nouri\\Desktop\\ORL_64x64_fea.CSV', header=None, skiprows=1)
#-----------



#-----------  
#  Set Main Label
#-----------
Labels = pd.read_csv('C:\\Users\\babak_Nouri\\Desktop\\ORL_64x64_gnd.CSV', header=None, skiprows=1)
Main_Labels=np.array(Labels).ravel().tolist()
#-----------



#-----------  
#  Set Steps For Count Of Update 
#-----------
Steps=30
#-----------



#-----------  
#  Set Aloha For MFFS  And SLSDR Algoritm
#-----------
Alpha=0.001
#-----------



#-----------  
#  Set Beta For SLSDR Algoritm
#-----------
Beta=0.001
#-----------



#-----------  
#  Set Lambda For SLSDR Algoritm
#-----------
Lambda=100
#-----------



#-----------  
#  Set Value for Break Algoritm
#-----------
Break_Value=4
#-----------



#-----------  
#  Set Number Of Dimension For 
#-----------
Dimension_Select_List=[20,30,40,50,60,70,80,90,100]
#-----------  



#-----------  
#  Set Count Of Send Each Dataset To Kmeans Algoritm
#-----------
Kmeans_Count=5
#-----------  



#-----------  
#  Set sigma For SLSDR Algoritm
#-----------
sigma=100
#-----------  



#-----------   
#  Call The Result Function
#-----------
Result(Main_Labels,Dimension_Select_List,Kmeans_Count,sigma)
#------------------------------------------------------


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




