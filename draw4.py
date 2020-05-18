
import matplotlib.pyplot as plt

X = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
Y = [0.64,0.744,0.761429,0.8033889,0.903,0.94,0.96,0.95,0.982355,1]
x = list(range(len(Y)))
total_width, n = 0.8, 2
width = total_width / n

plt.bar(x, Y, width=width, tick_label=X,label='Online Learning', fc='y')
plt.xlabel('Training time ')
plt.ylabel('Accuracy')
plt.legend()
plt.show()