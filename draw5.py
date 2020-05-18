import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#x = np.linspace(0, 2, 100)
x1 = [1,2,3,4]
y1=[0.80,0.854,0.92,0.98]
y2=[0.8967,0.906,0.9467,0.977]
y3=[0.79,0.88,0.935,0.99]
y4=[0.21,0.12,0.075,0.01]
plt.plot(x1, y1, marker='o',label='Precision',ls='-.')
plt.plot(x1, y2, marker='*',label='Recall',ls='-.')
plt.plot(x1, y3, marker='^',label='Accuracy',ls='-.')
plt.plot(x1, y4, marker='x',label='Error rate',ls='-.')
plt.xticks(x1)
plt.xlabel('Node communication radius ')
plt.ylabel('value')

#plt.title("Simple Plot")

plt.legend()

plt.show()
