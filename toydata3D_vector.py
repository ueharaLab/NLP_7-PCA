import numpy as np
from scipy.stats import multivariate_normal
import numpy as np
import japanize_matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams
#from mpl_toolkits.mplot3d import Axes3D
# 期待値と分散共分散行列の準備
def generate(mean):
    cov = np.array([[1,0,0],[0,0.1,0], [0,0,1]])
    data = np.random.multivariate_normal(mean, cov, size=100)
    dataset =[]
    for d in data:
        if len((d[d<0]))==0:
            dataset.append(d)

    return np.array(dataset)

mean = np.array([5,4,0.2])
syu = generate(mean)
mean = np.array([0.5,4,3])
prin = generate(mean)

rcParams['figure.figsize'] = 30,10

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

colors =['blue','red']

ax.scatter(syu[:,0],syu[:,1],syu[:,2], c=colors[0],s=100,alpha=0.3,label='シュークリーム')
ax.scatter(prin[:,0],prin[:,1],prin[:,2], c=colors[1],s=100,alpha=0.3,label='プリン')

y = 4
x = np.linspace(1,14, 11)
z = np.linspace(1,2, 11)
X, Z = np.meshgrid(x, z)
p = np.array([0.5, -0.5])
q = np.array([0.5, 2.5])
X = X * p[0] + Z * q[0] -1
Z = X * p[1] + Z * q[1] +0

Y = np.array([y] * X.shape[0])

#ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, alpha=0.3)


ax.quiver(3.5, 4, 2, 3, 0, -1.5,linewidth=2, length=1, arrow_length_ratio=0.1,color='blue' ) # 矢印
ax.quiver(3.5, 4, 2, 0.5, 0,1.5,linewidth=2, length=1, arrow_length_ratio=0.1,color='blue' ) # 矢印

ax.set_xlabel("カスタード")
ax.set_ylabel("生クリーム")
ax.set_zlabel("カラメル")
plt.legend()
ax.set_xlim(0,8)
ax.set_ylim(0,8)
ax.set_zlim(0,8)
plt.show()

# ---------------------------------------------------------------------------------

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

colors =['blue','red']

ax.scatter(syu[:,0],syu[:,1],syu[:,2], c=colors[0],s=100,alpha=0.3,label='シュークリーム')
ax.scatter(prin[:,0],prin[:,1],prin[:,2], c=colors[1],s=100,alpha=0.3,label='プリン')

y = 4
x = np.linspace(1,14, 11)
z = np.linspace(1,2, 11)
X, Z = np.meshgrid(x, z)
p = np.array([0.5, -0.5])
q = np.array([0.5, 2.5])
X = X * p[0] + Z * q[0] -1
Z = X * p[1] + Z * q[1] +0

Y = np.array([y] * X.shape[0])

#ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, alpha=0.3)


ax.quiver(3.5, 4, 2, 3, 0, -1.5,linewidth=2, length=1, arrow_length_ratio=0.1,color='blue' ) # 矢印
ax.quiver(3.5, 4, 2, 0.5, 0,1.5,linewidth=2, length=1, arrow_length_ratio=0.1,color='blue' ) # 矢印
ax.quiver(3.5, 4, 2, -0.2, -1.5,-0.1,linewidth=2, length=1, arrow_length_ratio=0.1,color='blue' ) # 矢印


ax.set_xlabel("カスタード")
ax.set_ylabel("生クリーム")
ax.set_zlabel("カラメル")
plt.legend()
ax.set_xlim(0,8)
ax.set_ylim(0,8)
ax.set_zlim(0,8)
plt.show()



