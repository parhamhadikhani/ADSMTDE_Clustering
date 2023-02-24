from random import random
from random import sample
from random import uniform
import numpy as np
from Evaluation import *
from sklearn import datasets
import time
import copy
from scipy.stats import cauchy
import random 
from scipy.spatial import distance as di
import warnings
#from GMM import *
#for calculatng diversity
import pandas as pd
from scipy.stats import entropy
#------------------------------

warnings.filterwarnings("ignore")
from numba import jit, cuda

# sum squre error
def calc_sse(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray):
    distances = 0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        dist = np.sum((data[idx] - c)**2)
        distances += dist
    return distances

@jit  
def calc_sse_Plus(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray):
    #distances = 0
    homo=[]
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        dist = np.sum((data[idx] - c)**2)
        homo.append(dist)
        #distances += dist
    return homo
@jit 
def createGrid(fitness):
    inflation=0.1
    f_max=np.max(np.array(fitness),axis=0)
    f_min=np.min(np.array(fitness),axis=0)  
    dc=f_max-f_min
    f_max=f_max+inflation*dc
    f_min=f_min-inflation*dc 
    mesh_id=np.zeros(len(fitness))   
    for i in range(len(fitness)):
        mesh_id[i]=_cal_mesh_id(fitness[i],f_min,f_max)
    unique_elements,counts_elements = np.unique(mesh_id,return_counts=True)
    counts_elements=10/(counts_elements**3) 
    prob=counts_elements/np.sum(counts_elements)
    idx=RouletteWheelSelection(prob)
    return idx

def _cal_mesh_id(fit,f_min,f_max):
    id_=0
    mesh_div=10
    for i in range(len(fit)):
        try:
            id_dim=int((fit[i]-f_min[i])*mesh_div/(f_max[i]-f_min[i]))
            id_ = id_ + id_dim*(mesh_div**i)
        except ValueError:
            id_dim=0
            id_ = id_ + id_dim*(mesh_div**i)
    return id_
@jit
def RouletteWheelSelection(prob):
    p=np.random.random()
    cunsum=0
    for i in range(len(prob)):
        cunsum+=prob[i]
        if p<=cunsum:
            return i
        
def Diameter(data,idx):
    #diameter = 0
    #idx = np.where(labels == i)
    temp1=data[idx]
    #print('temp1',len(temp1))
    dists = di.cdist(temp1, temp1, 'euclidean')
    #print('dists',len(dists))
    if len(dists)==0:
        return 0
    return np.max(dists)/2

 

def gaussian_kernel(distance, bandwidth):
    euclidean_distance = np.sqrt(((distance)**2).sum(axis=1))
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((euclidean_distance / bandwidth))**2)
    return val
# Calculate distance between data and centroids
@jit
def _calc_distance(data: np.ndarray,centroids) -> np.ndarray:
    distances = []
    for c in centroids:
        for i in range(len(data)):
            distances.append(np.linalg.norm(data[i,:,:] - c))
    distances = list(_divide_chunks(distances, len(data))) 
    distances = np.array(distances)
    distances = np.transpose(distances)
    return distances


def _divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n]

# Calculate distance between data and centroids       
def cdist_fast(XA, XB):

    XA_norm = np.sum(XA**2, axis=1)
    XB_norm = np.sum(XB**2, axis=1)
    XA_XB_T = np.dot(XA, XB.T)
    distances = XA_norm.reshape(-1,1) + XB_norm - 2*XA_XB_T
    return distances  
#Predict new data's cluster using minimum distance to centroid
def _predict(data, centroids):
    distance = cdist_fast(data,centroids)
    cluster = _assign_cluster(distance)
    return cluster

#Assign cluster to data based on minimum distance to centroids
def _assign_cluster(distance: np.ndarray):
    cluster = np.argmin(distance, axis=1)
    return cluster

class Vector:
    def __init__(
            self,
            n_cluster: int,
            data: np.ndarray):
        
        index = np.random.choice(list(range(len(data))), n_cluster)
        self.centroids = data[index].copy()
        self.fitness = calc_sse(self.centroids, _predict(data,self.centroids), data)
        
class LifeTime_archive:
    def __init__(self,vectors,size):
        self.size=size
        self.vectors=vectors
        self.lifetime=[x for x in self.vectors]
        
    def update(self,vector):
        self.lifetime.append(vector)
        while len(self.lifetime)>self.size:
            self.lifetime.pop(0)

    def select(self):
        select=np.random.randint(0,self.size)
        return self.lifetime[select].centroids
      
class differential_Evolution:

    def __init__(
            self,
            n_cluster: int,
            n_vectors: int,
            data: np.ndarray,
            max_iter: int = 100,
            mutate: float=0.5,
            recombination: float = 0.7,
            print_debug: int = 10):
        
        self.n_vectors=n_vectors
        self.n_cluster = n_cluster
        self.data = data
        self.max_iter = max_iter
        self.vectors = []
        self.lifetime=[]
        #self.data=data
        self.label=None
        self.gbest_sse = np.inf
        self.gbest_centroids = None
        self._init_vectors()
        self.mutate=mutate
        self.recombination=recombination
        self.print_debug = print_debug
        self.winTVPIdx = 1
        self.nFES =np.zeros((3,1))
        self.WinIter = 20
        self.nImpFit=np.ones((3,1))
        self.ImpRate=np.zeros((3,1))
        self.Mab=np.zeros((1,2))
        self.MAB_sse=0
        
    def _init_vectors(self):
        for i in range(self.n_vectors):

            vector = Vector(self.n_cluster, self.data)

            if vector.fitness < self.gbest_sse:
                self.gbest_centroids = vector.centroids.copy()
                self.gbest_sse = vector.fitness
                self.label=_predict(self.data,vector.centroids).copy()
            self.vectors.append(vector)
    
    
    
    def MUTATION(self,current):
        candidates = list(range(0,self.n_vectors))
        candidates.remove(current)
        random_index = sample(candidates, 3)
        v_donor=self.vectors[random_index[0]].centroids + self.mutate * (self.vectors[random_index[1]].centroids-self.vectors[random_index[2]].centroids)
        return v_donor
        
    def RECOMBINATION(self,v_donor,vector):
        v_trial=np.copy(vector.centroids)
        for k in range(len(v_trial)):
            crossover = random()
            if crossover <= self.recombination:
                v_trial[k]=v_donor[k]
        return v_trial
    
    def GREEDY_SELECTION(self,v_trial,vector,current):
        score_trial=calc_sse(v_trial, _predict(self.data,v_trial), self.data)

        if score_trial < vector.fitness:
            self.vectors[current].centroids = v_trial
            self.vectors[current].fitness=score_trial
    

    
    def distributing_population(self,i):
        if i!=0: 
            for w in range(3):
                self.ImpRate[w]=self.nImpFit[w]/self.nFES[w];
            self.winTVPIdx=np.argmax( self.ImpRate)
            self.nImpFit=np.ones((3,1))
            self.ImpRate=np.zeros((3,1))
            self.nFES =np.zeros((3,1))
        
        if self.winTVPIdx == 1:#R_tvp
            G_tvp= self.vectors[0:int(self.n_vectors*.2)]
            L_tvp = self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]
            R_tvp = self.vectors[int(2*0.2*self.n_vectors):]
            
        if self.winTVPIdx == 2:
            G_tvp= self.vectors[0:int(self.n_vectors*.2)]
            R_tvp = self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]
            L_tvp = self.vectors[int(2*0.2*self.n_vectors):]
        
        if self.winTVPIdx == 3:
            R_tvp= self.vectors[0:int(self.n_vectors*.2*2)]
            L_tvp = self.vectors[int(2*0.2*self.n_vectors):int(4*0.2*self.n_vectors)]
            G_tvp = self.vectors[int(4*0.2*self.n_vectors):]
            
        self.nFES = self.nFES + [[len(R_tvp)],[len(L_tvp)],[len(G_tvp)]]
        return R_tvp,L_tvp,G_tvp
    
    def Population_updating(self,G_tvp,R_tvp,L_tvp):
        if self.winTVPIdx == 1:
            
            self.vectors[0:int(self.n_vectors*.2)]=G_tvp
                
            
            self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]=L_tvp
                
                
            self.vectors[int(2*0.2*self.n_vectors):]=R_tvp
                
            
        if self.winTVPIdx == 2:
            
            self.vectors[0:int(self.n_vectors*.2)]=G_tvp
                
            
            self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]=R_tvp
                
                
            self.vectors[int(2*0.2*self.n_vectors):]=L_tvp
        
        if self.winTVPIdx == 3:
            
            self.vectors[0:int(self.n_vectors*.2*2)]=R_tvp
                
            
            self.vectors[int(2*0.2*self.n_vectors):int(4*0.2*self.n_vectors)]=L_tvp
                
                
            self.vectors[int(4*0.2*self.n_vectors):]=G_tvp
                                
    def R_TVP(self,r_tvp,ca1):
        index=[]
        #f=F[0:len(r_tvp)]
        M_tril=np.tril(np.ones((self.n_cluster,len(self.data[0,:]))))

        for vector in r_tvp:
            index.append(vector.fitness)
        #print('cluster:',vector.shape)
        best=r_tvp[np.argmin(index)]
        worst=r_tvp[np.argmax(index)]
        for i, vector in enumerate(r_tvp):
            M_tril=np.random.permutation(M_tril)
            M_bar=1-M_tril
            #print('cluster:',self.lifetime.shape)
            vi=vector.centroids +(best.centroids-vector.centroids)+(worst.centroids-vector.centroids)+ca1*(self.lifetime.select()-vector.centroids)
            ui=M_tril*vector.centroids + M_bar* vi
            score_trial=calc_sse(ui, _predict(self.data,ui), self.data)
            
            if score_trial < vector.fitness:
                self.lifetime.update(vector)
                r_tvp[i].centroids = ui
                r_tvp[i].fitness=score_trial
                self.nImpFit[0]+=1
        return r_tvp
    #random.choices(vectors, k=1)[0]
    def L_TVP(self,l_tvp,ca2):
        
        #f=F[0:len(l_tvp)]
        for i, vector in enumerate(l_tvp):
            #cauchy.cdf(vector.centroids)*
            
            ui=vector.centroids + (random.choices(l_tvp, k=1)[0].centroids-random.choices(l_tvp, k=1)[0].centroids)+ca2*(self.lifetime.select()-vector.centroids)
            score_trial=calc_sse(ui, _predict(self.data,ui), self.data)

            if score_trial < vector.fitness:
                self.lifetime.update(vector)
                l_tvp[i].centroids = ui
                l_tvp[i].fitness=score_trial
                self.nImpFit[1]+=1
        return l_tvp
    
    def G_TVP(self,g_tvp,ca2):
        M_tril=np.tril(np.ones((self.n_cluster,len(self.data[0,:]))))

        for i, vector in enumerate(g_tvp):
            #shift = []
            #ll=_predict(self.data,vector.centroids).copy()
            # for i, c in enumerate(vector.centroids):
            #     idx = np.where( ll == i)
            #     epsilon=Diameter(self.data,idx[0])
            #     if epsilon!=0:
            #         point_weights = gaussian_kernel(c-self.data[idx], epsilon)
            #         tiled_weights = np.tile(point_weights, [len(c), 1])
            #         # denominator
            #         denominator = sum(point_weights)
            #         shifted_point = np.multiply(tiled_weights.transpose(), self.data[idx]).sum(axis=0) / denominator
            #         shift.append(shifted_point)
            #     else:
            #         #print(epsilon,'++')
            #         shift.append(c)
                
            #shift=np.array(shift)
            #shift_center=(shift - vector.centroids)
            M_tril=np.random.permutation(M_tril)
            M_bar=1-M_tril
            vi = self.gbest_centroids + ca2*(random.choices(g_tvp, k=1)[0].centroids-random.choices(g_tvp, k=1)[0].centroids)
            ui=M_tril*vector.centroids + M_bar* vi
            score_trial=calc_sse(ui, _predict(self.data,ui), self.data)
            if score_trial < vector.fitness:
                #print('*1')
                # model = GMM(self.n_cluster, n_runs = 10)
                # a=model.fit_gmm(self.data, _predict(self.data,ui),ui)
                # l=model.predict_gmm(self.data)
                # gmmsocre=calc_sse(a.mu, l, self.data)
                # #print('*')
                # if gmmsocre<score_trial:
                #     #print('**')
                #     self.lifetime.update(vector)
                #     g_tvp[i].centroids = a.mu
                #     g_tvp[i].fitness=gmmsocre
                #     self.nImpFit[2]+=1
                # else:
                self.lifetime.update(vector)
                g_tvp[i].centroids = ui
                g_tvp[i].fitness=score_trial
                self.nImpFit[2]+=1
        return g_tvp
    
    
    def fit(self):
        #print('*555')
        self.lifetime=LifeTime_archive(self.vectors,self.n_vectors)
        memory_sf = 0.5 * np.ones((len(self.data[0,:]), 1))
        copy=np.mod(self.n_vectors,len(self.data[0,:]))
        history = []
        count=1
        MaxFES = len(self.data[0,:]) * 10000
        MaxGen = MaxFES/self.n_vectors;
        Gen=0
        Mu = np.log(len(self.data[0,:]))
        initial = 0.001
        final = 2
        #The winner-based distributing substep
       
            
        entropy_R_tvp=[]
        entropy_L_tvp=[]
        entropy_G_tvp=[]
        entropyt=[]
        
        for i in range(self.max_iter):
        # #DE
        #     for vector in self.vectors:
        #         v_donor=self.MUTATION(i)
        #         v_trial=self.RECOMBINATION(v_donor,vector)
        #         self.GREEDY_SELECTION(v_trial,vector,i)
        #     for vector in self.vectors:
        #         if vector.fitness < self.gbest_sse:
        #             self.gbest_centroids = vector.centroids.copy()
        #             self.gbest_sse = vector.fitness
        #             self.label=_predict(self.data,vector.centroids).copy()
        #     history.append(self.gbest_sse)
        # return history,self.label,self.gbest_sse
        # #/DE

        ##MTDE
            diversity_R_tvp=[]
            diversity_L_tvp=[]
            diversity_G_tvp=[]
            
            Gen = Gen +1
            ca1 = 2 - Gen * ((2) /MaxGen)                                        
            ca2 = (initial - (initial - final) * (((MaxGen - Gen)/MaxGen))**Mu)
            


            #===========================The winner-based distributing substep================
            if i!=0:
                for w in range(3):
                    self.ImpRate[w]=self.nImpFit[w]/self.nFES[w];
                self.winTVPIdx=np.argmax( self.ImpRate)
                self.nImpFit=np.ones((3,1))
                self.ImpRate=np.zeros((3,1))
                self.nFES =np.zeros((3,1))

            if self.winTVPIdx == 1:#R_tvp
                G_tvp= self.vectors[0:int(self.n_vectors*.2)]
                L_tvp = self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]
                R_tvp = self.vectors[int(2*0.2*self.n_vectors):]
                for ii in range(len(R_tvp)):    
                    diversity_R_tvp.append(R_tvp[ii].centroids)
                for ii in range(len(L_tvp)):
                    diversity_L_tvp.append(L_tvp[ii].centroids)
                for ii in range(len(G_tvp)):
                    diversity_G_tvp.append(G_tvp[ii].centroids)
                pd_series = pd.Series(diversity_R_tvp)
                counts = pd_series.value_counts()
                entropy_R_tvp.append( entropy(counts))
                pd_series = pd.Series(diversity_L_tvp)
                counts = pd_series.value_counts()
                entropy_L_tvp.append( entropy(counts))
                pd_series = pd.Series(diversity_G_tvp)
                counts = pd_series.value_counts()
                entropy_G_tvp.append( entropy(counts))
                
            if self.winTVPIdx == 2:
                G_tvp= self.vectors[0:int(self.n_vectors*.2)]
                R_tvp = self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]
                L_tvp = self.vectors[int(2*0.2*self.n_vectors):]
                for ii in range(len(R_tvp)):    
                    diversity_R_tvp.append(R_tvp[ii].centroids)
                for ii in range(len(L_tvp)):
                    diversity_L_tvp.append(L_tvp[ii].centroids)
                for ii in range(len(G_tvp)):
                    diversity_G_tvp.append(G_tvp[ii].centroids)
                pd_series = pd.Series(diversity_R_tvp)
                counts = pd_series.value_counts()
                entropy_R_tvp.append( entropy(counts))
                pd_series = pd.Series(diversity_L_tvp)
                counts = pd_series.value_counts()
                entropy_L_tvp.append( entropy(counts))
                pd_series = pd.Series(diversity_G_tvp)
                counts = pd_series.value_counts()
                entropy_G_tvp.append( entropy(counts))
            
            if self.winTVPIdx == 3:
                R_tvp= self.vectors[0:int(self.n_vectors*.2*2)]
                L_tvp = self.vectors[int(2*0.2*self.n_vectors):int(4*0.2*self.n_vectors)]
                G_tvp = self.vectors[int(4*0.2*self.n_vectors):]
                for ii in range(len(R_tvp)):    
                    diversity_R_tvp.append(R_tvp[ii].centroids)
                for ii in range(len(L_tvp)):
                    diversity_L_tvp.append(L_tvp[ii].centroids)
                for ii in range(len(G_tvp)):
                    diversity_G_tvp.append(G_tvp[ii].centroids)
                pd_series = pd.Series(diversity_R_tvp)
                counts = pd_series.value_counts()
                entropy_R_tvp.append( entropy(counts))
                pd_series = pd.Series(diversity_L_tvp)
                counts = pd_series.value_counts()
                entropy_L_tvp.append( entropy(counts))
                pd_series = pd.Series(diversity_G_tvp)
                counts = pd_series.value_counts()
                entropy_G_tvp.append( entropy(counts))
                
                
            self.nFES = self.nFES + [[len(R_tvp)],[len(L_tvp)],[len(G_tvp)]]
            
            
            #r_tvp,l_tvp,g_tvp=self.distributing_population(i)
            #print('vectors:',len(self.vectors))
            #===========================R-TVP=====================================
            #print('beforlen(R_tvp)',len(R_tvp))
            R_tvp=self.R_TVP(R_tvp,ca1)
            #print('afterlen(R_tvp)',len(R_tvp))
            #===========================L-TVP=====================================
            #print('beforlen(R_tvp)',len(R_tvp))

            L_tvp=self.L_TVP(L_tvp,ca2)
            #print('afterlen(R_tvp)',len(R_tvp))
            #===========================G-TVP=====================================
            #print('beforlen(R_tvp)',len(R_tvp))

            G_tvp=self.G_TVP(G_tvp,ca2)
            #print('afterlen(R_tvp)',len(R_tvp))
            #print('G_TVP',len(g_tvp),'R_TVP',len(r_tvp),'L_TVP',len(l_tvp),'***')
            #======================= Population updating =========================================
            
            self.Population_updating(G_tvp,R_tvp,L_tvp)
            #print(' self.vectors',len( self.vectors))
            for vector in self.vectors:
                if vector.fitness < self.gbest_sse:
                    self.gbest_centroids = vector.centroids.copy()
                    self.gbest_sse = vector.fitness
                    self.label=_predict(self.data,vector.centroids).copy()
            history.append(self.gbest_sse)
            if i % self.print_debug == 0:
                print('Iteration {:04d}/{:04d} current gbest score {:.18f}'.format(
                    i + 1, self.max_iter, self.gbest_sse))
        print('Finish with gbest score {:.18f}'.format(self.gbest_sse))
        return history,self.label,self.gbest_sse,entropy_R_tvp,entropy_L_tvp,entropy_G_tvp
        ##/MTDE
 
        
    
 
    def run(self):
        data_star=np.copy(self.data)
        cen=[]
        Mintpt=len(self.data)/self.n_cluster
        cluster_label=np.zeros(len(self.data))
        cluster_label = cluster_label.astype(int)
        incremental=0
        
        indexes=np.zeros((len(self.data),2))
        indexes[:,0]=np.arange(start=0, stop=len(self.data), step=1)
        indexes[:,1]=np.arange(start=0, stop=len(self.data), step=1)
        h=[]
        
        entropy_R_tvpk=[]
        entropy_L_tvpk=[]
        entropy_G_tvpk=[]
        while(self.n_cluster>=2):
            print('self.n_cluster',self.n_cluster)
            if len(self.data)!=0:
                self.vectors = []
                self.winTVPIdx = 1
                self.nFES =np.zeros((3,1))
                self.WinIter = 20
                self.nImpFit=np.ones((3,1))
                self.ImpRate=np.zeros((3,1))
                self.label=None
                self.gbest_sse = np.inf
                self.gbest_centroids = None
                self.lifetime=[]
                self._init_vectors()
                
                history,label,S,entropy_R_tvp,entropy_L_tvp,entropy_G_tvp=self.fit()
                # for i in range(self.max_iter):
                h.append(history)
                entropy_R_tvpk.append(entropy_R_tvp)
                entropy_L_tvpk.append(entropy_L_tvp)
                entropy_G_tvpk.append(entropy_G_tvp)
                    #clustering
                
                    
                #find the global best
                #calcaulate the homegenity of all clusters
                f=calc_sse_Plus(self.gbest_centroids, self.label, self.data)
                idx=np.where(self.label == np.argmin(f))
                #achieve the tghe id of the cluster
                #idcenter=createGrid(f)
                #collect the member of that cluster
                #print("Parham Hadikhani")
                
                #idx=np.where(self.label == idcenter)
                print('numbre of member',len(idx[0]))
                print('mintpt',Mintpt)
                if len(idx[0])>=Mintpt:
                    #print ('finish6')
                    #self.history.append(self.gbest_sse)
                    c=self.gbest_centroids[np.argmin(f)]
                    #c=self.gbest_centroids[idcenter]
                    center=c.reshape(1,len(c))
                    index=np.copy(idx[0])
                    indx=np.copy(index)
                    
                    for i in range(len(index)):
                        for j in range(len(indexes)):
                            if index[i]==indexes[j,1]:
                                index[i]=int(indexes[j,0])
                                break
                    
                    cluster_label[index]=incremental
                    self.data = np.delete(self.data, indx, axis=0)
                    indexes=np.delete(indexes, indx, axis=0)
                    
                    indexes[:,1]=np.arange(start=0, stop=len(self.data), step=1)
                    cen.append(center)
                    self.n_cluster-=1
                    incremental+=1
                    print('new ',self.n_cluster)
                    #self.data=np.copy(data_star)
                else:
                    self.n_cluster-=1
                    print('decrese',self.n_cluster)
            else:
                break
        if len(self.data)!=0:
            last_centroid=np.mean(self.data,axis=0).reshape(1,len(self.data[0,:]))
            cen.append(last_centroid)
            idx=[]
            for i in range(len(indexes)):
                idx.append(int(indexes[i,0]))

            cluster_label[idx]=incremental
            cen=np.array(cen)
        cen=np.array(cen)
        centroids = cen.reshape(cen.shape[0], (cen.shape[1]*cen.shape[2]))
        #obj1,obj2=Multi_Objectvies(centroids, cluster_label,self.data)
        calc_sse(centroids, cluster_label, data_star)
        #calc_sse(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray):
        return h,cluster_label,S,centroids,entropy_R_tvpk,entropy_L_tvpk,entropy_G_tvpk
    
         



