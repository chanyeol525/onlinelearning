import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#x = np.linspace(0, 2, 100)
x1 = [0.1,0.3,0.5,0.7]

y1=[0.8533,0.8895,0.838889,0.7785]
y2=[0.95,0.915,0.855,0.801667]
y3=[0.15,0.125,0.175,0.236]
y4=[0.05,0.085,0.185,0.20]
plt.plot(x1, y1, marker='o',color="red", label='Accuracy of OGD')
plt.plot(x1, y2, marker='*',color="blue",label='Accuracy of PA')
plt.plot(x1, y3, marker='^',color="red",label='Error rate of OGD')
plt.plot(x1, y4, marker='x',color="blue",label='Error rate of PA')
plt.xticks(x1)
plt.xlabel('percentage of malicious nodes')
plt.ylabel('value')

#plt.title("Simple Plot")

plt.legend()

plt.show()
plt.savefig("1Draw.png")
