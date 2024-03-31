# 主成分平面上へのデータプロット
1. PC1,PC2をx,y軸とする主成分平面上にデータを射影する。  
2. 以下の変数transformedの列方向は何を意味するか
3. 寄与率explained_variance_ratio_の要素数はいくつか
4. バイプロットのベクトル座標は、主成分ベクトル行列の転置で求められる。pca.components_.T
### 演習
以下のプログラムを修正してPC3,PC4をx,y軸とする主成分平面上にデータをプロットせよ。バイプロットもPC3,PC4の座標をプロットせよ。

[toydata_pca.py](toydata_pca.py)


``` python
dataset_df = pd.read_csv('./data/toydata.csv', index_col=0,encoding='ms932', sep=',',skiprows=0)
dataset_df.sample(frac=1, random_state=0)

dataset = dataset_df.values
feature_names = dataset_df.columns 
cluster = dataset_df.index


fig, ax = plt.subplots(1, 1, figsize=(6, 4))

pca = PCA(n_components=4)	
pca_iris1=pca.fit(dataset)
# pca 空間上にデータを射影。主成分空間は4次元。
# 主成分得点は4次元だが、PC1,PC2がデータの分散方向を良くとらえているので、その平面上にプロットする、
transformed = pca.transform(dataset)


colors={'シュークリーム':'red','プリン':'green','杏仁豆腐':'blue'}
syu = transformed[cluster=='シュークリーム']
ax.scatter(syu[:,0],syu[:,1], c=colors['シュークリーム'],s=100,alpha=0.5,label='シュークリーム')
prin = transformed[cluster=='プリン']
ax.scatter(prin[:,0],prin[:,1], c=colors['プリン'],s=100,alpha=0.5,label='プリン')
anin = transformed[cluster=='杏仁豆腐']
ax.scatter(anin[:,0],anin[:,1], c=colors['杏仁豆腐'],s=100,alpha=0.5,label='杏仁豆腐')
'''
pca_vectors=pca.components_.T #バイプロット
clist = ['pink','brown','orange','purple']
for i,(vector,feature_name) in enumerate(zip(pca_vectors,feature_names)):
	ax.arrow(0,0,vector[0]*4,vector[1]*4,width=0.05,head_width=0.1,head_length=0.1,length_includes_head=True,color=clist[i])
	ax.annotate(feature_name,xy=(vector[0]*4,vector[1]*4),size=16,color = clist[i])
'''	
# 寄与率を計算
contrib_list=np.round(pca.explained_variance_ratio_, decimals=3)

ax.set_xlabel('PC1 ({})'.format(contrib_list[0]),fontsize=18)
ax.set_ylabel('PC2 ({})'.format(contrib_list[1]),fontsize=18)
ax.set_title('主成分平面',fontsize=18)

plt.legend(bbox_to_anchor=(1,0), loc='lower right')
plt.show()
```