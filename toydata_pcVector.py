import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
import pandas as pd
import japanize_matplotlib



dataset_df = pd.read_csv('./data/toydata.csv', index_col=0,encoding='ms932', sep=',',skiprows=0)
dataset_df.sample(frac=1, random_state=0)

dataset = dataset_df.values
feature_names = dataset_df.columns 
cls = dataset_df.index

fig, ax = plt.subplots(1, 1, figsize=(6, 4))


colors={'シュークリーム':'red','プリン':'green','杏仁豆腐':'blue'}
syu = dataset[cls=='シュークリーム']
ax.scatter(syu[:,0],syu[:,2], c=colors['シュークリーム'],s=100,alpha=0.5,label='シュークリーム')
prin = dataset[cls=='プリン']
ax.scatter(prin[:,0],prin[:,2], c=colors['プリン'],s=100,alpha=0.5,label='プリン')
anin = dataset[cls=='杏仁豆腐']
ax.scatter(anin[:,0],anin[:,2], c=colors['杏仁豆腐'],s=100,alpha=0.5,label='杏仁豆腐')



ax.set_xlabel(feature_names[0],fontsize=14)
ax.set_ylabel(feature_names[2],fontsize=14)               

pca = PCA(n_components=4)
pca.fit(dataset)

pca_vectors = pd.DataFrame(pca.components_,index=['PC1','PC2','PC3','PC4'],columns=feature_names)
print(pca_vectors)
colorlist = ['pink','brown','orange','purple']
labels = ['PC1','PC2','PC3','PC4']
for vector1,c,l in zip(pca.components_,colorlist,labels):
	
    ax.arrow(0,0,vector1[0]*4,vector1[2]*4,width=0.05,head_width=0.1,head_length=0.1,length_includes_head=True,color=c)
    ax.annotate(l,xy=(vector1[0]*4,vector1[2]*4),size=16,color = c)

plt.legend(bbox_to_anchor=(1,0), loc='lower right')
plt.show()

