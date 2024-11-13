
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
cov_matrix = np.cov(dataset.T) # 共分散行列のmethod　データセットを転置する
fig, ax = plt.subplots(1,1, figsize=(8,8))

# -------  主成分ベクトル（固有ベクトル）、寄与率（固有値）を計算して降順にソート
eigen_val,pca_vec = np.linalg.eig(cov_matrix) #pca_vec は縦ベクトルで出力されることに注意！https://analytics-note.xyz/programming/matrix-eigenvalues/

eigs = np.concatenate([eigen_val.reshape(-1,1), pca_vec.T],axis=1)
eigs = eigs.real#複素数の実部を取り出す
idx_sort = np.argsort(eigs[:, 0])[::-1]# 固有値の降順に並べたインデックスを取り出す
eigs_sorted = eigs[idx_sort,:]#インデックスで降順ソート
eig_vecs_sorted = eigs_sorted[:,1:]
eig_vals_sorted = eigs_sorted[:,0]

# -------- 主成分得点を計算

num_vecs = 2 # 取り出したい主成分（第1主成分、第2主成分。。。）数
pca_score = np.dot(eig_vecs_sorted[:num_vecs,:], dataset.T)# 主成分得点行列を求める　行方向：主成分　列方向：データ方向
pca_score = pca_score.T


pca_score_df=pd.DataFrame(pca_score,columns=np.arange(pca_score.shape[1]).tolist())
pca_score_df = pd.concat([csv_input.iloc[:,0],pca_score_df],axis=1)#ラベル付き主成分得点の表
colors=['orange','red','blue','gold']

sns.scatterplot(x=0, y=1, hue='keyword', palette="hls",data=pca_score_df,s=50,alpha=0.4)

# --------- 主成分負荷ベクトル（空間軸の主成分平面への射影）を求める

vocab_e = np.eye(len(feature_names))
pca_loading_matrix = np.dot(eig_vecs_sorted[:num_vecs,:], vocab_e)#主成分負荷量の行列（データの次元分のone hotベクトルの行列がvocab_e)を主成分ベクトルに射影したときの長さをと求める


p_norm = {name:np.linalg.norm(pca_loading_matrix[:,i], ord=2) for i,name in enumerate(feature_names) }#データの次元毎に主成分負荷量のノルムを計算
p_norm_sorted = dict(sorted(p_norm.items(), key=lambda x:x[1],reverse=True))# 主成分負荷量ノルムの降順にソート（データ空間の軸の主成分ベクトル（ノルム１）に対する射影
print(p_norm_sorted)
ranking =25
top_ranked_voc = [name for i,name in enumerate(p_norm_sorted) if i < ranking]#上記ノルム　トップランクを取り出す
p_matrix_df = pd.DataFrame(pca_loading_matrix,columns=feature_names)

top_vocs = p_matrix_df.loc[:,top_ranked_voc]# 主成分負荷行列からトップランクの次元だけ取り出す

top_vocs = top_vocs.T #行方向：主成分負荷ベクトルの数（トップランク）　列方向：主成分ベクトルの数
print(top_vocs)
for name,row in top_vocs.iterrows():
    vals = row.values
    ax.arrow(0,0,vals[0]*1,vals[1]*1,width=0.001,head_width=0.005,head_length=0.01,length_includes_head=True,color='blue')
    ax.annotate(name,xy=(vals[0]*1,vals[1]*1),size=16,color = 'black')

     
contrib_list=np.round(eig_vals_sorted, decimals=3)
print(contrib_list)
ax.set_xlabel('PC1 ({})'.format(contrib_list[0]),fontsize=18)
ax.set_ylabel('PC2 ({})'.format(contrib_list[1]),fontsize=18)
plt.show()


    
        
    
