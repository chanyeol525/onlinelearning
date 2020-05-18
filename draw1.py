import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#x = np.linspace(0, 2, 100)
x1 = [5,15,25,35]
y1=[0.84,0.89,0.959,0.967]
y2=[0.8,0.85,0.8625,0.90]
y3=[0.78,0.815,0.87,0.91]
y4=[0.22,0.19,0.13,0.09]
plt.plot(x1, y1, marker='o',label='Precision',ls='-.')
plt.plot(x1, y2, marker='*',label='Recall', ls='-.')
plt.plot(x1, y3, marker='^',label='Accuracy', ls='-.')
plt.plot(x1, y4, marker='x',label='Error rate', ls='-.')
plt.xticks(x1)
plt.xlabel('Number of nodes')
plt.ylabel('value')

#plt.title("Simple Plot")

plt.legend()

plt.show()
plt.savefig("1Draw.png")
