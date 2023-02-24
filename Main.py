import numpy as np
from Evaluation import *
#from sklearn import datasets
import time
from DE import differential_Evolution
import umap
from Autoencoder import autoencoder
#from torch.autograd import Variable
from torch.utils.data import DataLoader
from Dataset import load_mnist
import torch
from torch import nn
from sklearn import metrics
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics.cluster import adjusted_rand_score
import torch.nn.functional as F
import csv

_matrix=[]
_sse=[]
_accuracy=[]
_NMI=[]
_cluster_accuracy=[]
_adjusted_rand_score=[]
_label=[]
_bestsse=[]
_time=[]
_homogeneity_score=[]
_completeness_score=[]
_v_measure_score=[]
_fowlkes_mallows_score=[]
_precision_score=[]
_recall_score=[]
_estimated_k=[]



def sparse_loss( images,model_children):
    loss = 0
    values = images
    for i in range(len(model_children[0])):
        values = F.relu((model_children[0][i](values)))
        loss += torch.mean(torch.abs(values))
    return loss  

use_sparse = False
RHO = 0.01
BETA = 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data, true_label ,tag= load_mnist()
#data, true_label ,tag= load_har()
#data, true_label,tag = load_usps()
#data, true_label ,tag= load_pendigits()
#data, true_label ,tag=load_fashion()
data=torch.tensor(data)
true_label=torch.tensor(true_label)

epoch_loss = []
num_epochs = 100
net = autoencoder(numLayers=[data[0].view(-1).shape[0], 500, 500, 2000, 10]).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
train_loader = DataLoader(dataset=data,
	                batch_size=512, 
	                shuffle=False)
criterion = nn.MSELoss()
model_children = list(net.children())
train_loss = []
for epoch in range(num_epochs):
    loss_train = 0.0
    net.train()
    for data in train_loader:

        img  = data.float()
        img = img.to(device=device)
        output = net(img)
        output = output.view(output.size(0), 28*28)
        mse_loss = criterion(output, img)

        if use_sparse:

            #sparsity = sparse_loss(RHO, img,model_children)
            l1_loss = sparse_loss(img,model_children)
            # add the sparsity penalty
            #print('sparse')
            #loss = mse_loss + BETA * sparsity
            loss = mse_loss + BETA * l1_loss
        else:
            loss = mse_loss
        #losses.append(loss)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss.append(loss_train / (epoch+1))
    print('epoch [{}/{}], MSE_loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

encoder = nn.Sequential(*[net.layers[i] for i in range(7)])
encoder.to(device)


data, true_label ,tag= load_mnist()


lbl=true_label.cpu().numpy()
k=int(np.sqrt(len(data))/4)
features = []
abbas=[]
train_loader = DataLoader(dataset=data,
	                batch_size=len(data), 
	                shuffle=False)
for i, batch in enumerate(train_loader):
	    img = batch.float()
	    img = img.to(device)
	    abbas.append(img.detach().cpu())
	    features.append(encoder(img).detach().cpu())
features = torch.cat(features)
abbas=torch.cat(abbas)



hl = features.cpu().data.numpy()
data = umap.UMAP(
    random_state=0,
    metric='euclidean',
    n_components=2,
    n_neighbors=10,
    min_dist=0).fit_transform(hl)
#folder='7june2022'
#type1='MNIST'
folder='RESULTS'
dataset=['MNIST','har','MNIST','MNIST','MNIST']
type1=dataset[0]
savefile1=r'./%s'%folder+'/%s'%type1 +'/result.csv'
savefile2=r'./%s'%folder+'\%s'%type1 +'/fscore.csv'
savefile3=r'./%s'%folder+'\%s'%type1 +'/number_k_sub1%d.csv'

matcon=r'./%s'%folder+'\%s'%type1 +'/confusion.npy'
timesave=r'./%s'%folder+'\%s'%type1 +'/time.npy'
history1plot=r'./%s'%folder+'\%s'%type1 +'/history.npy'
ll=r'./%s'%folder+'/%s'%type1 +'/label.npy'
lossplot=r'./%s'%folder+'/%s'%type1 +'/loss1.npy'



for i in range(30):

    start = time.time()   
    de = differential_Evolution(n_cluster=k, n_vectors=600, data=data)
    history,label,S,centroids=de.run()
    _estimated_k.append(len(centroids))
    #estimated_k1.append(len(np.unique(label)))
    end = time.time()
    runtime=end - start
    _time.append(runtime)
    
    _bestsse.append(S)
    _sse.append(history)
    label=map2(lbl,label)
    _label.append(label)
    _matrix.append(confusion_matrix(lbl, label))
    _NMI.append(NMI(lbl, label))
    _accuracy.append(accuracy(lbl, label))
    _cluster_accuracy.append(cluster_accuracy(lbl, label))
    _adjusted_rand_score.append(adjusted_rand_score(lbl, label))
    _homogeneity_score.append(metrics.homogeneity_score(lbl, label))
    _completeness_score.append(metrics.completeness_score(lbl, label))
    _v_measure_score.append(metrics.v_measure_score(lbl, label))
    _fowlkes_mallows_score.append(metrics.fowlkes_mallows_score(lbl, label))
    _precision_score.append(precision_score(lbl, label, average='micro'))
    _recall_score.append(recall_score(lbl, label, average='micro'))
    
    print('accuracy: ',accuracy(lbl, label))
    print(i,'--------------------------------------------------------------') 
   
print('\n')
print('accuracy: ','Max: ',np.max(_accuracy),'Min: ',np.min(_accuracy),'_Mean: ',np.sum(_accuracy)/len(_accuracy))
print('\n')
print('NMI: ','Max: ',np.max(_NMI),'Min: ',np.min(_NMI),'_Mean: ',np.sum(_NMI)/len(_NMI))
np.save(ll,_label[np.argmax(_accuracy)])
_time=np.array(_time)

np.save(timesave,_time)
np.save(history1plot,_sse[np.argmax(_accuracy)])
np.save(lossplot,np.array(epoch_loss))

np.save(matcon,_matrix[np.argmax(_accuracy)])




myFile = open(savefile1, 'w')
with myFile:    
    myFields = ['metric','Mean', 'Min','Max']
    writer = csv.DictWriter(myFile, fieldnames=myFields)    
    writer.writeheader()
    writer.writerow({'metric':'sse','Mean': np.sum(_bestsse)/len(_bestsse), 'Min': np.min(_bestsse),'Max':np.max(_bestsse)})
    writer.writerow({'metric':'accuracy','Mean':np.sum(_accuracy)/len(_accuracy),'Min':np.min(_accuracy),'Max':np.max(_accuracy)})
    writer.writerow({'metric':'k ','Mean':round(np.sum(_estimated_k)/30),'Min':_estimated_k[np.argmin(_accuracy)],'Max':_estimated_k[np.argmax(_accuracy)]})
    writer.writerow({'metric':'NMI','Mean':np.sum(_NMI)/len(_NMI),'Min':np.min(_NMI),'Max':np.max(_NMI)})
    writer.writerow({'metric':'precision_score','Mean':np.sum(_precision_score)/len(_precision_score),'Min':np.min(_precision_score),'Max': np.max(_precision_score)})
    writer.writerow({'metric':'recall_score','Mean':np.sum(_recall_score)/len(_recall_score),'Min':np.min(_recall_score),'Max': np.max(_recall_score)})
    writer.writerow({'metric':'v_measure_score','Mean':np.sum(_v_measure_score)/len(_v_measure_score),'Min':np.min(_v_measure_score),'Max': np.max(_v_measure_score)})
    writer.writerow({'metric':'completeness_score','Mean':np.sum(_completeness_score)/len(_completeness_score),'Min':np.min(_completeness_score),'Max': np.max(_completeness_score)})
    writer.writerow({'metric':'homogeneity_score','Mean':np.sum(_homogeneity_score)/len(_homogeneity_score),'Min':np.min(_homogeneity_score),'Max':np.max(_homogeneity_score)})
    writer.writerow({'metric':'adjusted_rand_score','Mean':np.sum(_adjusted_rand_score)/len(_adjusted_rand_score),'Min':np.min(_adjusted_rand_score),'Max':np.max(_adjusted_rand_score)})
    writer.writerow({'metric':'fowlkes_mallows_score','Mean':np.sum(_fowlkes_mallows_score)/len(_fowlkes_mallows_score),'Min':np.min(_fowlkes_mallows_score),'Max': np.max(_fowlkes_mallows_score)})

ind=[]
for i in range(len(np.unique(lbl))):
    idx=np.where(lbl==i)
    ind.append(len(idx[0]))
ind=np.array(ind)
flist=F_Score(ind,lbl,_label)

f = open(savefile2, 'w')
a=['0','1','2','3','4','5','6','7','8','9']
with f:
    writer = csv.writer(f)    
    writer.writerow(a)
    writer.writerow(flist)

file = open(savefile3, 'w')

with file:
    writer = csv.writer(file)
    writer.writerow(_estimated_k)




   
    
 

    
    
    
    
    
