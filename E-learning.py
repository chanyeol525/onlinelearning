import numpy as np

from Graph import Vertex, Edge, Graph
#from Perceptron import Perceptron




from OGD import OGD
from Node import Node
from Package import Package
import random
import math
from sklearn.cluster import KMeans
from GraphInfoMaker import GraphInfoMaker
from Utils import Utils

#建图start
GI = GraphInfoMaker()
GI.generate()
print('节点个数:')
print(len(GI.getVertexSet()))

print('边个数:')
print(len(GI.getEdgeSet()))

VS = GI.getVertexSet()
ES = GI.getEdgeSet()
#print(VS)
g = Graph(VS, ES)

pathSet = g.find_all_paths('S', 'D', [])
'''for path in pathSet:
    print(path)'''
print('地图路径条数为:')
#print(len(pathSet))
#建图end
#路径的逆置
print("路径的逆置")
pathRe = pathSet

for path in pathSet:
    for i in range(0, int(len(path)/2)):
        node = path[i]
        path[i] = path[len(path) - 1-i]
        path[len(path) - 1 - i] = node

#print(pathSet)

#根据百分比去除一些路径start
useRate = 1
reduceRate = 1 - useRate
reduceSet = random.sample(range(0, len(pathSet)), int(len(pathSet)*reduceRate))

newPathSet = []
for i in range(0, len(pathSet)):
    if i not in reduceSet:
        newPathSet.append(pathSet[i])

print('经过修改多样性后的地图路径条数为(使用率 '+str(useRate)+' ):')
pathSet = newPathSet
#print(len(pathSet))
#根据百分比去除一些路径end

#建立点的概率信息start
#确定哪几个节点为恶意节点start
maliciousRate = 0.3#恶意节点比例
maliciousCount = int(len(GI.getVertexSet()) * maliciousRate) #恶意节点数量
maliciousSet = []
while len(maliciousSet) < maliciousCount:
    randomIndex = random.randint(0, len(pathSet)-1)
    randomNodeIndex = random.randint(0, len(pathSet[randomIndex]) - 1)

    if pathSet[randomIndex][randomNodeIndex] != 'S' and pathSet[randomIndex][randomNodeIndex] != 'D' \
            and pathSet[randomIndex][randomNodeIndex] not in maliciousSet:
        maliciousSet.append(pathSet[randomIndex][randomNodeIndex])
#确定哪几个节点为恶意节点end

print('')
print('恶意节点为:')
print(maliciousSet)
print('恶意节点数量:')
print(len(maliciousSet))

NS = {}

for v in VS:
    if v.getID() in maliciousSet:
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
#print(pathRe)
#从D注包，由各个顶点返回，逆置的路径首个节点都是D（sink）
count=0
newpathset = []
newpath = []
for pack in SS:
    newpath = []
    index = random.randint(0, len(pathRe) - 1)  # 离散均匀随机数
    path = pathRe[index]
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
#print("全新路径")
#print(newpathset)
#感知器start
#X = np.array([[3, 2, 1], [1, 1, 1], [0, 2, 1]])
#X = np.array([list(gg)])
X = None

#输出
#Y = np.array([10, 6, 7])
Y = None
Y1=None
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
        inputX[node] = 1
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

    if len(reputation["receiveSet"]) == 0:
        successRate=0
    else:
        successRate = len(reputation["receiveSet"]) / len(reputation["sendSet"])
        s1=len(reputation["receiveSet"]) / len(reputation["sendSet"])
        successRate = math.log(successRate)

    if Y1 is None:
        Y1 = np.array([s1])
    else:
        addY = np.array([s1])
        Y1 = np.r_[Y1, addY]

    if Y is None:
        Y = np.array([successRate])
    else:
        addY = np.array([successRate])
        Y = np.r_[Y, addY]
#整理方程式的输入end
print("感知器输入X")
#print(X)
print("感知器输入Y")
print(Y1)

#gradientDescent=gradientDescent()
#trustValue=gradientDescent.gradientDescent(X,Y)
#SGD=SGD()
#trustValue = SGD.partial_fit(X,Y)#取回信任值
#m,n=np.shape(X)
'''FTRLs=FTRLs(10000)
xTrans = X.transpose()  # 矩阵转置
trustValue = FTRLs.train(xTrans, Y,100000) #取回信任值'''
#FTRL = FTRL(n)
#trustValue = FTRL.ftrl(X, Y) #取回信任值
ogd=OGD(X.shape[1],alpha=0.05)
trustValue=ogd.OGD_(X,Y)
print("信任值")
print(trustValue)

#根据信任度聚类start
x1 = np.array(list(trustValue))
x1 = x1.reshape(-1, 1)
print('聚类信任值：')
print(x1)

#输出以下不进行加强检测的结果start
print('OGD')
print('未进行加强的聚类结果')
kmeans = KMeans(2)
kmeans.fit(x1)
print("聚类结果（标签）:")
print(kmeans.labels_)
print("聚类中心:")
print(kmeans.cluster_centers_)

BG = []
MG = []
center0 = kmeans.cluster_centers_[0][0]
center1 = kmeans.cluster_centers_[1][0]

benignIndex = 0
maliciousIndex = 0
if center0 <= center1:
    maliciousIndex = 0
    benignIndex = 1
elif center1 <= center0:
    maliciousIndex = 1
    benignIndex = 0

for i in range(0, len(kmeans.labels_)):
    if kmeans.labels_[i] == benignIndex:
        BG.append('R' + str(i))
    elif kmeans.labels_[i] == maliciousIndex:
        MG.append('R' + str(i))

print('BG(未加强)')
print(BG)
print('MG(未加强)')
print(MG)
allset=[]
for v in GI.getVertexSet():
    allset.append(v.getID())

leftSet=[v for v in allset if v not in maliciousSet]
#print(maliciousSet)
#print(leftSet)

TP = [v for v in maliciousSet if v in MG]
TN=[v for v in leftSet if v in BG]

FN=[v for v in maliciousSet if v in BG]
FP=[v for v in leftSet if v in MG]


print('precesion ：')
p=float(len(TP))/float(len(TP)+len(FP))
print(p)
print('recall：')
r=float(len(TP))/float(len(TP)+len(FN))
print(r)
print('Accuracy：')
a=float(len(TP)+len(TN))/float(len(GI.getVertexSet())-2)
print(a)

print('false：')
fa=float(len(FP)+len(FN))/float(len(GI.getVertexSet())-2)
print(fa)

print('识别准确度：')
inter = [v for v in maliciousSet if v in MG]
ar=float(len(inter))/float(len(maliciousSet))
print(ar)
print('识别错误率')
if len(MG)>len(inter):
  f=float((len(MG)-len(inter)))/float((len(GI.getVertexSet())-2-len(maliciousSet)))
else:
    f=0
print(f)

#输出以下不进行加强检测的结果end

kmeans = KMeans(3)
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
        BG.append('R' + str(i))
    elif kmeans.labels_[i] == unknownIndex:
        UG.append('R' + str(i))
    elif kmeans.labels_[i] == maliciousIndex:
        MG.append('R' + str(i))

print('BG')
print(BG)
print('UG')
print(UG)
print('MG')
print(MG)
######################################以下为增强检测###############################
EPS = [] #增强检测路径集

for unknownNode in UG:
    PS = []

    for path in newpathset:
        if unknownNode in path:
            PS.append(path)#统计具有未确认节点的路径

    #未知节点离散度与恶意节点离散度计算
    DDInfo = []
    for i in range(0, len(PS)):
        DDMG = Utils.calSimilarity(PS[i], MG)  #路径的恶意节点离散度
        DDUG = Utils.calSimilarity(PS[i], UG)  #路径的未知节点离散度
        length = len(PS[i])

        DDInfo.append({
            'index':i,
            'DDMG': DDMG,
            'DDUG': DDUG,
            'length': length
        })

    #判断是否存在DDMG = 0
    flag = 0
    for i in range(0, len(DDInfo)):
        if DDInfo[i]['DDMG'] == 0:
            flag = 1
            break

    if flag == 1:
        #存在DDMG为0的路径
        #先计算最小的DDUG
        minDDUG = 100000
        for info in DDInfo:
            if info['DDMG'] == 0 and info['DDUG'] < minDDUG:
                minDDUG = info['DDUG']

        S1 = []
        for info in DDInfo:
            if info['DDMG'] == 0 and info['DDUG'] == minDDUG:
                S1.append(info)

        minLength = 100000
        for info in S1:
            if info['length'] < minLength:
                minLength = info['length']

        for info in S1:
            if info['length'] == minLength:
                if PS[info['index']] not in EPS:
                    EPS.append(PS[info['index']])

    else:
        #不存在DDMG为0的路径
        #先计算最小的DDMG
        minDDMG = 100000
        for info in DDInfo:
            if info['DDMG'] < minDDMG:
                minDDMG = info['DDMG']

        S2 = []
        for info in DDInfo:
            if info['DDMG'] == minDDMG:
                S2.append(info)

        minDDUG = 100000
        #计算最小的DDUG
        for info in S2:
            if info['DDUG'] < minDDUG:
                minDDUG = info['DDUG']

        S3 = []
        for info in S2:
            if info['DDUG'] == minDDUG:
                S3.append(info)

        minLength = 100000
        for info in S3:
            if info['length'] < minLength:
                minLength = info['length']

        for info in S3:
            if   info['length'] == minLength:
                if PS[info['index']] not in EPS:
                    EPS.append(PS[info['index']])

#至此，EPS构造完成
#print(EPS)
#接下来往EPS注包，收集数据#####################################################
#生成包流start
SS = []

for i in range(0, 100000):
    pack = Package(str(i), "N")
    SS.append(pack)
#生成包流end

#生成接收集start
ReputationSet = []
#生成接收集end

#从D注包，由各个顶点返回，逆置的路径首个节点都是D（sink）
count=0
newpathset = []
newpath = []
for pack in SS:
    newpath = []
    index = random.randint(0, len(EPS) - 1)  # 离散均匀随机数
    path = EPS[index]
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

#整理方程式的输入start
for reputation in ReputationSet:
    if len(reputation["sendSet"]) == 0:
        continue # 说明这条路径废弃，没有任何包从这条路径上流过
    if len(reputation["receiveSet"]) == 0:
        continue  # 说明这条路径废弃，这条路是断路

    inputX = []
    for i in range(0, len(VS)-2):#减去S和D
        inputX.append(0)

    pathIndex = reputation["pathIndex"]
    path = newpathset[pathIndex]

    for i in range(1, len(path)-1):
        node = path[i]
        node = int(node.strip("R"))#将Ri的R去掉，得到i
        inputX[node] = 1

    # 增加X到方程当中
    if X is None:
        X = np.array([list(inputX)])
    else:
        addX = np.array([list(inputX)])
        X = np.r_[X, addX]

    # 增加Y到方程当中
    successRate = len(reputation["receiveSet"]) / len(reputation["sendSet"])

    successRate = math.log(successRate)

    if Y is None:
        Y = np.array([successRate])
    else:
        addY = np.array([successRate])
        Y = np.r_[Y, addY]
#整理方程式的输入end

#perceptron = Perceptron()

print('')
print("以下为增强检测结果:")
print("增强检测信任值:")
m,n=np.shape(X)
#trustValue = FTRL.ftrl(X, Y) #取回信任值
#trustValue=ogd.OGD_(X,Y)
#trustValue=gradientDescent.gradientDescent(X,Y)
#xTrans = X.transpose()  # 矩阵转置
ogd=OGD(X.shape[1],alpha=0.05)
trustValue=ogd.OGD_(X,Y)
#trustValue = FTRLs.train(xTrans, Y,100000) #取回信任值
#trustValue = perceptron.fit(X, Y) #取回信任值
for i in range(0, len(trustValue)):
    if trustValue[i] > 1.0:
        trustValue[i] = 1.0
print("信任值:")
print(trustValue)

#根据信任度聚类start
x1 = np.array(list(trustValue))
x1 = x1.reshape(-1, 1)
print('聚类信任值：')
print(x1)

kmeans = KMeans(2)
kmeans.fit(x1)
print("聚类结果（标签）:")
print(kmeans.labels_)
print("聚类中心:")
print(kmeans.cluster_centers_)
#根据信任度聚类end

#划分出BG,UG和MG
FBG = []
FMG = []

center0 = kmeans.cluster_centers_[0][0]
center1 = kmeans.cluster_centers_[1][0]

benignIndex = 0
maliciousIndex = 0
if center0 <= center1:
    maliciousIndex = 0
    benignIndex = 1
elif center1 <= center0:
    maliciousIndex = 1
    benignIndex = 0

for i in range(0, len(kmeans.labels_)):
    if kmeans.labels_[i] == benignIndex:
        FBG.append('R' + str(i))
    elif kmeans.labels_[i] == maliciousIndex:
        FMG.append('R' + str(i))

print('FBG')
print(FBG)
print('FMG')
print(FMG)
TP = [v for v in maliciousSet if v in FMG]
TN=[v for v in leftSet if v in FBG]

FN=[v for v in maliciousSet if v in FBG]
FP=[v for v in leftSet if v in FMG]

print('precesion ：')
p=float(len(TP))/float(len(TP)+len(FP))
print(p)
print('recall：')
r=float(len(TP))/float(len(TP)+len(FN))
print(r)
print('Accuracy：')
a=float(len(TP)+len(TN))/float(len(GI.getVertexSet())-2)
print(a)

print('false：')
fa=float(len(FP)+len(FN))/float(len(GI.getVertexSet())-2)
print(fa)


print('识别准确度：')
inter = [v for v in maliciousSet if v in FMG]
ar=len(inter)/len(maliciousSet)
print(ar)
print('识别错误率')
if len(FMG)>len(inter):
  f=(len(FMG)-len(inter))/(len(GI.getVertexSet())-2-len(maliciousSet))
else:
    f=0
print(f)
'''#######################接下来是HD的结果###########################
#统计所有的信任值
#生成结果集start
HDResultSet = []
for i in range(0, len(GI.getVertexSet()) - 2):
    hdResult = {
        'index': i, #节点编号
        'detectCount': 0, #出现次数
        'trust': 0.0 #信任值
    }
    HDResultSet.append(hdResult)
#生成结果集end

for reputation in ReputationSet:
    if len(reputation["sendSet"]) == 0:
        continue  # 说明这条路径废弃，没有任何包从这条路径上流过
    if len(reputation["receiveSet"]) == 0:
        continue  # 说明这条路径废弃，这条路是断路

    pathIndex = reputation["pathIndex"]
    path = newpathset[pathIndex]

    successRate = len(reputation["receiveSet"]) / len(reputation["sendSet"])
    trust = math.pow(successRate, 1.0/len(path))

    for i in range(1, len(path)-1):
        node = path[i]
        node = int(node.strip("R"))#将Ri的R去掉，得到i
        HDResultSet[node]["detectCount"] = HDResultSet[node]["detectCount"] + 1
        HDResultSet[node]["trust"] = HDResultSet[node]["trust"] + trust

for hdResult in HDResultSet:
    if hdResult["detectCount"] > 0:
        hdResult["trust"] = hdResult["trust"] / hdResult["detectCount"]
    else:
        #说明包没有流经这个节点或不可判定
        hdResult["trust"] = 1.0

HDTrustValue = []
for i in range(0, len(HDResultSet)):
    HDTrustValue.append(HDResultSet[i]["trust"])

print('')
print("以下为HD检测结果:")
print("HD检测信任值:")
print(HDTrustValue)

#根据信任度聚类start
x1 = np.array(list(HDTrustValue))
x1 = x1.reshape(-1, 1)
print('聚类信任值：')
print(x1)

kmeans = KMeans(2)
kmeans.fit(x1)
print("聚类结果（标签）:")
print(kmeans.labels_)
print("聚类中心:")
print(kmeans.cluster_centers_)
#根据信任度聚类end

BG = []
MG = []
center0 = kmeans.cluster_centers_[0][0]
center1 = kmeans.cluster_centers_[1][0]

benignIndex = 0
maliciousIndex = 0
if center0 <= center1:
    maliciousIndex = 0
    benignIndex = 1
elif center1 <= center0:
    maliciousIndex = 1
    benignIndex = 0

for i in range(0, len(kmeans.labels_)):
    if kmeans.labels_[i] == benignIndex:
        BG.append('R' + str(i))
    elif kmeans.labels_[i] == maliciousIndex:
        MG.append('R' + str(i))

print('BG(HD)')
print(BG)
print('MG(HD)')
print(MG)

######################################以下为增强检测###############################
EPS = [] #增强检测路径集

for unknownNode in UG:
    PS = []

    for path in newpathset:
        if unknownNode in path:
            PS.append(path)#统计具有未确认节点的路径

    #未知节点离散度与恶意节点离散度计算
    DDInfo = []
    for i in range(0, len(PS)):
        DDMG = Utils.calSimilarity(PS[i], MG)  #路径的恶意节点离散度
        DDUG = Utils.calSimilarity(PS[i], UG)  #路径的未知节点离散度
        length = len(PS[i])

        DDInfo.append({
            'index':i,
            'DDMG': DDMG,
            'DDUG': DDUG,
            'length': length
        })

    #判断是否存在DDMG = 0
    flag = 0
    for i in range(0, len(DDInfo)):
        if DDInfo[i]['DDMG'] == 0:
            flag = 1
            break

    if flag == 1:
        #存在DDMG为0的路径
        #先计算最小的DDUG
        minDDUG = 100000
        for info in DDInfo:
            if info['DDMG'] == 0 and info['DDUG'] < minDDUG:
                minDDUG = info['DDUG']

        S1 = []
        for info in DDInfo:
            if info['DDMG'] == 0 and info['DDUG'] == minDDUG:
                S1.append(info)

        minLength = 100000
        for info in S1:
            if info['length'] < minLength:
                minLength = info['length']

        for info in S1:
            if info['length'] == minLength:
                if PS[info['index']] not in EPS:
                    EPS.append(PS[info['index']])

    else:
        #不存在DDMG为0的路径
        #先计算最小的DDMG
        minDDMG = 100000
        for info in DDInfo:
            if info['DDMG'] < minDDMG:
                minDDMG = info['DDMG']

        S2 = []
        for info in DDInfo:
            if info['DDMG'] == minDDMG:
                S2.append(info)

        minDDUG = 100000
        #计算最小的DDUG
        for info in S2:
            if info['DDUG'] < minDDUG:
                minDDUG = info['DDUG']

        S3 = []
        for info in S2:
            if info['DDUG'] == minDDUG:
                S3.append(info)

        minLength = 100000
        for info in S3:
            if info['length'] < minLength:
                minLength = info['length']

        for info in S3:
            if info['length'] == minLength:
                if PS[info['index']] not in EPS:
                    EPS.append(PS[info['index']])

#至此，EPS构造完成
#接下来往EPS注包，收集数据#####################################################
#生成包流start
SS = []

for i in range(0, 100000):
    pack = Package(str(i), "N")
    SS.append(pack)
#生成包流end

#生成接收集start
ReputationSet = []
#生成接收集end

#从D注包，由各个顶点返回，逆置的路径首个节点都是D（sink）
count=0
newpathset = []
newpath = []
for pack in SS:
    newpath = []
    index = random.randint(0, len(EPS) - 1)  # 离散均匀随机数
    path = EPS[index]
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

#整理方程式的输入start
for reputation in ReputationSet:
    if len(reputation["sendSet"]) == 0:
        continue # 说明这条路径废弃，没有任何包从这条路径上流过
    if len(reputation["receiveSet"]) == 0:
        continue  # 说明这条路径废弃，这条路是断路

    inputX = []
    for i in range(0, len(VS)-2):#减去S和D
        inputX.append(0)

    pathIndex = reputation["pathIndex"]
    path = newpathset[pathIndex]

    for i in range(1, len(path)-1):
        node = path[i]
        node = int(node.strip("R"))#将Ri的R去掉，得到i
        inputX[node] = 1

    # 增加X到方程当中
    if X is None:
        X = np.array([list(inputX)])
    else:
        addX = np.array([list(inputX)])
        X = np.r_[X, addX]

    # 增加Y到方程当中
    successRate = len(reputation["receiveSet"]) / len(reputation["sendSet"])

    successRate = math.log(successRate)

    if Y is None:
        Y = np.array([successRate])
    else:
        addY = np.array([successRate])
        Y = np.r_[Y, addY]
#整理方程式的输入end

#perceptron = Perceptron()

print('')
print("以下为增强检测结果:")
print("增强检测信任值:")

#trustValue=gradientDescent.gradientDescent(X,Y)
xTrans = X.transpose()  # 矩阵转置
trustValue = FTRLs.train(xTrans, Y,10000) #取回信任值
#trustValue = perceptron.fit(X, Y) #取回信任值
for i in range(0, len(trustValue)):
    if trustValue[i] > 1.0:
        trustValue[i] = 1.0
print("信任值:")
print(trustValue)

#根据信任度聚类start
x1 = np.array(list(trustValue))
x1 = x1.reshape(-1, 1)
print('聚类信任值：')
print(x1)

kmeans = KMeans(2)
kmeans.fit(x1)
print("聚类结果（标签）:")
print(kmeans.labels_)
print("聚类中心:")
print(kmeans.cluster_centers_)
#根据信任度聚类end

#划分出BG,UG和MG
FBG = []
FMG = []

center0 = kmeans.cluster_centers_[0][0]
center1 = kmeans.cluster_centers_[1][0]

benignIndex = 0
maliciousIndex = 0
if center0 <= center1:
    maliciousIndex = 0
    benignIndex = 1
elif center1 <= center0:
    maliciousIndex = 1
    benignIndex = 0

for i in range(0, len(kmeans.labels_)):
    if kmeans.labels_[i] == benignIndex:
        FBG.append('R' + str(i))
    elif kmeans.labels_[i] == maliciousIndex:
        FMG.append('R' + str(i))

print('FBG')
print(FBG)
print('FMG')
print(FMG)
'''