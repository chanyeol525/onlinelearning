import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#x = np.linspace(0, 2, 100)
x1 = [0.1,0.3,0.5,0.7]
y1=[0.804,0.964,0.90,0.85]
y2=[0.86667,0.916667,0.8467,0.83333]
y3=[0.965,0.915,0.875,0.815]
y4=[0.035,0.085,0.135,0.195]
plt.plot(x1, y1, marker='o',label='Precision',ls='-.')
plt.plot(x1, y2, marker='*',label='Recall',ls='-.')
plt.plot(x1, y3, marker='^',label='Accuracy',ls='-.')
plt.plot(x1, y4, marker='x',label='Error rate',ls='-.')
plt.xticks(x1)
plt.xlabel('probability of attack ')
plt.ylabel('value')

#plt.title("Simple Plot")

plt.legend()

plt.show()
plt.savefig("1Draw.png")
