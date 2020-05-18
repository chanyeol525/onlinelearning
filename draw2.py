import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#x = np.linspace(0, 2, 100)
x1 = [0.1,0.3,0.5,0.7]
y1=[0.9,0.95,0.86,0.84]
y2=[0.9,0.99,0.89,0.9]
y3=[0.95,0.915,0.815,0.831667]
y4=[0.05,0.085,0.185,0.179]
plt.plot(x1, y1, marker='o',label='Precision',ls='-.')
plt.plot(x1, y2, marker='*',label='Recall',ls='-.')
plt.plot(x1, y3, marker='^',label='Accuracy',ls='-.')
plt.plot(x1, y4, marker='x',label='Error rate',ls='-.')
plt.xticks(x1)
plt.xlabel('percentage of malicious nodes')
plt.ylabel('value')

#plt.title("Simple Plot")

plt.legend()

plt.show()
plt.savefig("1Draw.png")
