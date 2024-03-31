
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import japanize_matplotlib

dataset_df = pd.read_csv('./data/toydata.csv', index_col=0,encoding='ms932', sep=',',skiprows=0)
dataset_df.sample(frac=1, random_state=0)

dataset = dataset_df.values
feature_names = dataset_df.columns 
cluster = dataset_df.index


fig, ax = plt.subplots(1, 1, figsize=(6, 4))


# 主成分分析クラスのインスタンス化(n_components:計算する主成分の数）
pca = PCA(n_components=4)	
pca_iris1=pca.fit(dataset)
transformed = pca.transform(dataset)


colors={'シュークリーム':'red','プリン':'green','杏仁豆腐':'blue'}
syu = transformed[cluster=='シュークリーム']
ax.scatter(syu[:,0],syu[:,1], c=colors['シュークリーム'],s=100,alpha=0.5,label='シュークリーム')
prin = transformed[cluster=='プリン']
ax.scatter(prin[:,0],prin[:,1], c=colors['プリン'],s=100,alpha=0.5,label='プリン')
anin = transformed[cluster=='杏仁豆腐']
ax.scatter(anin[:,0],anin[:,1], c=colors['杏仁豆腐'],s=100,alpha=0.5,label='杏仁豆腐')
print(pca.components_.T)
'''
pca_vectors=pca.components_.T
clist = ['pink','brown','orange','purple']
for i,(vector,feature_name) in enumerate(zip(pca_vectors,feature_names)):
	ax.arrow(0,0,vector[0]*4,vector[1]*4,width=0.05,head_width=0.1,head_length=0.1,length_includes_head=True,color=clist[i])
	ax.annotate(feature_name,xy=(vector[0]*4,vector[1]*4),size=16,color = clist[i])
'''
contrib_list=np.round(pca.explained_variance_ratio_, decimals=3)
print(contrib_list)
ax.set_xlabel('PC1 ({})'.format(contrib_list[0]),fontsize=18)
ax.set_ylabel('PC2 ({})'.format(contrib_list[1]),fontsize=18)
ax.set_title('主成分平面',fontsize=18)

plt.legend(bbox_to_anchor=(1,0), loc='lower right')
plt.show()


	
		
	
