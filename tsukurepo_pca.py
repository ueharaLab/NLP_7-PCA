
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
import pandas as pd
import japanize_matplotlib
import seaborn as sns



csv_input = pd.read_csv('./data/tsukurepo_bow_pca.csv', encoding='ms932', sep=',',skiprows=0)
#素性をTF-IDFにするなら、create_dataset_tfidfを使う
cluster=csv_input['keyword'].tolist()
cls = {c:i for i,c in enumerate(list(set(cluster)))}
#print(cls)

dataset=csv_input.iloc[:,4:].values
feature_names = csv_input.iloc[:,4:].columns
fig, ax = plt.subplots(1,1, figsize=(8,8))

# 主成分分析クラスのインスタンス化(n_components:計算する主成分の数）
pca = PCA(n_components=130)	
pca.fit(dataset)
transformed = pca.transform(dataset)
print(transformed)
tr_df = pd.DataFrame(transformed)
#cls_df = pd.DataFrame(,columns=['dish'])
pca_score_df = pd.concat([csv_input.iloc[:,0],tr_df],axis=1)
print(pca_score_df)

colors=['orange','red','blue','gold']

sns.scatterplot(x=0, y=1, hue='keyword', palette="hls",data=pca_score_df,s=200,alpha=0.4)
'''
for i,(c,t) in enumerate(zip(cluster,transformed)):	

	ax.scatter(t[0],t[1],c=colors[cls[c]])
	
	#ax.legend(("ティラミス', 'ミルフィーユ', 'プリン', 'タルトタタン"),loc="upper left")
'''
pca_vectors=pca.components_.T[:,:2]
pca_vec_sqsum = [(pv[0]**2) + (pv[1]**2) for pv in pca_vectors]
pca_vecs=[]
for pca_vec, sq_sum, feature in zip(pca_vectors, pca_vec_sqsum, feature_names):
    	pca_vecs.append([sq_sum,pca_vec[0],pca_vec[1],feature])

pca_vecs_df = pd.DataFrame(pca_vecs,columns=['sq_sum','x','y','name'])
pca_vecs_df = pca_vecs_df.sort_values('sq_sum', ascending=False)


for i,row in pca_vecs_df.iloc[:25,:].iterrows():
	ax.arrow(0,0,row['x']*1,row['y']*1,width=0.001,head_width=0.005,head_length=0.01,length_includes_head=True,color='blue')
	#ax.quiver(0,0,row['x'],row['y'],angles='xy',scale_units='xy',scale=1)
	ax.annotate(row['name'],xy=(row['x']*1,row['y']*1),size=16,color = 'black')
		
contrib_list=np.round(pca.explained_variance_ratio_, decimals=3)

ax.set_xlabel('PC1 ({})'.format(contrib_list[0]),fontsize=18)
ax.set_ylabel('PC2 ({})'.format(contrib_list[1]),fontsize=18)
plt.show()


	
		
	
