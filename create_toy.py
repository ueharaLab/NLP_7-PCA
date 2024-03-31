import numpy as np
import pandas as pd
import codecs
def generate(mean,datasize,name):    
	cov = np.array([[1,0,0,0],[0,0.1,0,0], [0,0,1,0],[0,0,0,1]])
	data = np.random.multivariate_normal(mean, cov, size=datasize)
	dataset =[]
	for d in data:
		if len((d[d<0]))==0:
			dataset.append(d)						
	t = [name]*len(dataset)
	return np.array(dataset),t

datasize = 100
mean = np.array([5,4,0.2,0.1])
target_name ='シュークリーム'
syu,syu_t = generate(mean,datasize,target_name)
mean = np.array([0.5,4,3,0.1])
target_name = 'プリン'
prin,prin_t = generate(mean,datasize,target_name)
mean = np.array([0.1,4,0.1,2])
target_name = '杏仁豆腐'
anin,anin_t = generate(mean,datasize,target_name)

feature_names = ['カスタード','生クリーム','カラメル','ゼラチン']
target_names = ['シュークリーム','プリン','杏仁豆腐']

all_data = np.concatenate([syu,prin,anin])
all_t = syu_t + prin_t + anin_t
print(len(all_t))
print(all_data.shape)
dataset_df = pd.DataFrame(all_data, index=all_t, columns=feature_names)

with codecs.open("./data/toydata.csv", "w", "ms932", "ignore") as f: 
    #header=Trueで、見出しを書き出す
    dataset_df.to_csv(f, index=True, encoding="ms932", mode='w', header=True)