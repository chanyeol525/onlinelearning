import numpy as np

from Graph import Vertex, Edge, Graph
from Perceptron import Perceptron
from Node import Node
from Package import Package
import random
import math
from sklearn.cluster import KMeans

#建图start
S = Vertex('S')
R1 = Vertex('R1')
R2 = Vertex('R2')
R3 = Vertex('R3')
R4 = Vertex('R4')
D = Vertex('D')

e0 = Edge(S, R1)
e1 = Edge(S, R2)
e2 = Edge(S, R3)
e3 = Edge(R1, R2)
e4 = Edge(R3, R2)
e5 = Edge(R2, R4)
e6 = Edge(R3, R4)
e7 = Edge(R1, D)
e8 = Edge(R2, D)
e9 = Edge(R4, D)


VS = [S, R1, R2, R3, R4, D]
ES = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9]

g = Graph(VS, ES)
print(g)
print("正常路径")
pathSet = g.find_all_paths(S.getID(), D.getID(), [])
for path in pathSet:
    print(path)
#建图end

#路径的逆置
#print("路径的逆置")
#pathRe=pathSet

#for path in pathSet:
  #  for i in range(0, int(len(path)/2)):
    #    node=path[i]
    #    path[i]=path[len(path) - 1-i]
    #    path[len(path) - 1 - i]=node
#print(pathRe)
#建立点的概率信息start

NS = {}

for v in VS:
    if v.getID() == 'R4':
        node = Node(v.getID(), 0.3, 0.0, 0.0)
    else:
        node = Node(v.getID(), 0.0, 0.0, 0.0)

    NS[v.getID()] = node
#建立点的概率信息end

#生成包流start
SS = []

for i in range(0, 10000):
    pack = Package(str(i), "N")
    SS.append(pack)
#生成包流end

#生成接收集start
ReputationSet = []
'''for i in range(0, len(pathSet)):
    reputation = {
        'pathIndex': i, #路径编号
        'sendSet': [],
        'receiveSet': []
    }
    ReputationSet.append(reputation)
'''
#生成接收集end

#从D注包，由各个顶点返回，逆置的路径首个节点都是D（sink）
count=0
newpathset = []
newpath = []
for pack in SS:
    newpath = []
    index = random.randint(0, len(path) - 1)  # 离散均匀随机数
    path = path[index]
    for i in range(1, len(path) - 1):
        newpath = newpath + [path[i]]
        if newpath not in newpathset:
            newpathset.append(newpath)
            reputation = {
                'pathIndex': count,  # 路径编号
                'sendSet': [],
                'receiveSet': []
            }
            ReputationSet.append(reputation)
            count = count+1
        pathIndex = newpathset.index(newpath)

        ReputationSet[pathIndex]['sendSet'].append(pack)
        pTA = NS[path[i]].get_pTA()
        pDA = NS[path[i]].get_pDA()
        pRA = NS[path[i]].get_pRA()
        pN = 1 - pTA - pDA - pRA
        p = random.random()
        if p <= pN:
            pack.setFlag("N") # 正常，不作修改，成功传到下一个节点
        else:
            pack.setFlag("D")  # 代表损坏
        # 各个节点按路径返回包
        for j in range(len(newpath) - 1, -1, -1):
            pTA = NS[path[j]].get_pTA()
            pDA = NS[path[j]].get_pDA()
            pRA = NS[path[j]].get_pRA()
            pN = 1 - pTA - pDA - pRA
            p = random.random()
            if p <= pN:
                continue
            else:
                pack.setFlag("D")
                break
        if pack.getFlag() == "N":
            ReputationSet[pathIndex]['receiveSet'].append(pack)
print("全新路径")
print(newpathset)
'''
#注包及处理start
for pack in SS:
    index = random.randint(0, len(pathSet)-1) #离散均匀随机数
    ReputationSet[index]['sendSet'].append(pack)
    path = pathSet[index]


    for i in range(1, len(path)-1):
        #node 拿出来是ID
        pTA = NS[path[i]].get_pTA()
        pDA = NS[path[i]].get_pDA()
        pRA = NS[path[i]].get_pRA()
        pN = 1 - pTA - pDA - pRA

        p = random.random()

        if p <= pN:
            continue #正常，不作修改，成功传到下一个节点
        else:
            pack.setFlag("D") #代表损坏
            break

    if pack.getFlag() == "N":
        ReputationSet[index]['receiveSet'].append(pack)
#注包及处理end
'''
'''for reputation in ReputationSet:
    print("pathIndex:")
    print(reputation["pathIndex"])
    print("sendNumber:")
    print(len(reputation["sendSet"]))
    print("receiveNumber:")
    print(len(reputation["receiveSet"]))
    print("")'''

#感知器start
#X = np.array([[3, 2, 1], [1, 1, 1], [0, 2, 1]])
#X = np.array([list(gg)])
X = None

#输出
#Y = np.array([10, 6, 7])
Y = None

#整理方程式的输入start
for reputation in ReputationSet:
    inputX = []
    for i in range(0, len(VS)-2):#减去S和D
        inputX.append(0)

    pathIndex = reputation["pathIndex"]
    path = newpathset[pathIndex]
  # print(inputX)
   # print("reputation 路径")
  #  print(path)

    for i in range(0, len(path)):
        node = path[i]
        node = int(node.strip("R"))#将Ri的R去掉，得到i
        inputX[node-1] = 1
    #print(inputX)
    # 增加X到方程当中
    if X is None:
        X = np.array([list(inputX)])
    else:
        addX = np.array([list(inputX)])   #存储单一数据类型的多维数组
        X = np.r_[X, addX]             #np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
                                       #np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。

    # 增加Y到方程当中
    if len(reputation["sendSet"]) == 0:
        continue #说明这条路径废弃，没有任何包从这条路径上流过
    print("receiveSet")
    print(len(reputation["receiveSet"]))
    print("sendSet")
    print(len(reputation["sendSet"]))
    successRate = len(reputation["receiveSet"]) / len(reputation["sendSet"])
    if len(reputation["receiveSet"]) == 0:
        successRate=0
    else:
        successRate = len(reputation["receiveSet"]) / len(reputation["sendSet"])
        successRate = math.log(successRate)

    if Y is None:
        Y = np.array([successRate])
    else:
        addY = np.array([successRate])
        Y = np.r_[Y, addY]
#整理方程式的输入end

print(X)
print(Y)

#输入数据
'''
2X1 + X2 = 4, 3X1 + 2X2 = 7
'''

perceptron = Perceptron()

trustValue = perceptron.fit(X, Y) #取回信任值
print("信任值:")
print(trustValue)
#感知器end

#根据信任度聚类start
x1 = np.array(list(trustValue))
x1 = x1.reshape(-1, 1)
print('聚类信任值：')
print(x1)

kmeans=KMeans(3)
kmeans.fit(x1)
print("聚类结果（标签）:")
print(kmeans.labels_)
print("聚类中心:")
print(kmeans.cluster_centers_)
#根据信任度聚类end

#划分出BG,UG和MG
BG = []
UG = []
MG = []

center0 = kmeans.cluster_centers_[0][0]
center1 = kmeans.cluster_centers_[1][0]
center2 = kmeans.cluster_centers_[2][0]

benignIndex = 0
unknownIndex = 0
maliciousIndex = 0
if center0 <= center1 <= center2:
    maliciousIndex = 0
    unknownIndex = 1
    benignIndex = 2
elif center0 <= center2 <= center1:
    maliciousIndex = 0
    unknownIndex = 2
    benignIndex = 1
elif center1 <= center0 <= center2:
    maliciousIndex = 1
    unknownIndex = 0
    benignIndex = 2
elif center1 <= center2 <= center0:
    maliciousIndex = 1
    unknownIndex = 2
    benignIndex = 0
elif center2 <= center0 <= center1:
    maliciousIndex = 2
    unknownIndex = 0
    benignIndex = 1
elif center2 <= center1 <= center0:
    maliciousIndex = 2
    unknownIndex = 1
    benignIndex = 0

for i in range(0, len(kmeans.labels_)):
    if kmeans.labels_[i] == benignIndex:
        BG.append('R' + str(i+1))
    elif kmeans.labels_[i] == unknownIndex:
        UG.append('R' + str(i+1))
    elif kmeans.labels_[i] == maliciousIndex:
        MG.append('R' + str(i+1))

print('BG')
print(BG)
print('UG')
print(UG)
print('MG')
print(MG)
