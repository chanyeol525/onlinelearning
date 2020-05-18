#!/etc/bin/python
#coding=utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#np.random.seed(2000)
#y = np.random.standard_normal((10, 2))

x1 = [5,15,25,35]
y1=[1,0.9014,0.869,0.867]
y2=[0.9,0.76,0.7625,0.67]
y3=[0.96,0.88,0.87,0.86]
y4=[0.04,0.12,0.124,0.137]
plt.figure(figsize=(7,5))
plt.plot(x1,y1,marker='o',label = 'Precision')
plt.plot(x1,y2,marker='*', label = 'Recall')
plt.plot(x1,y3,marker='0', label = 'Accuracy')
plt.plot(x1,y4,marker='*', label = 'Error rate')
#plt.grid(True)


y =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
plt.xticks(x1)
plt.yticks(y)
#plt.axis('tight')
#是设置坐标轴显示范围为紧凑型 matlab画图会根据画图的数据范围自动调整坐标轴的范围 使得显示的图像或者曲线可以全部显示出来
plt.xlabel('Number of nodes')
plt.ylabel('value')
plt.legend() #图例位置自动
plt.show()