from Graph import Vertex, Edge, Graph
from Node import Node
import random
import os
import math
import numpy
#10*10的方格，放50个节点，保证节点与sink之间都联通
#[]表示list数据类型

class GraphInfoMaker:
    areaWidth = 10#宽度
    vertexCount =5#要生成多少个点
  #  r=2   #黑洞半径
  #  n=1 #黑洞个数
    vertexSet = [] #点集
    edgeSet = [] #边集
    map = [] #地图


    def getVertexSet(self):
        return self.vertexSet

    def getEdgeSet(self):
        return self.edgeSet

    def generate(self):
        #先生成地图，记录某些格子里已经放置了多少个点
        for i in range(0, self.areaWidth):
            self.map.append([])#放行   append() 方法用于在列表末尾添加新的对象。
        for i in range(0, self.areaWidth):
            for j in range(0, self.areaWidth):
                self.map[i].append(0)

        flag = 0
        while flag == 0:
            #恢复一些记录参数start
            self.vertexSet.clear()
            self.edgeSet.clear()
            for i in range(0, self.areaWidth):
                for j in range(0, self.areaWidth):
                    self.map[i][j] = 0

            maxNumberInColumn = int(math.sqrt(self.vertexCount)) #一列最多放置的节点个数
            column = 0 #当前放到第几列
            placedNumber = 0 #当前列已经放了多少个节点
            # 恢复一些记录参数end

#不均匀网络，放洞（圆心，半径之内无法放点）

      #      Cx=random.randint(0,self.areaWidth)
      #      Cy=random.randint(0,self.areaWidth)
            for i in range(0, self.vertexCount):
                # 先生成点在的格子坐标
                emptyLoc = []#空余位置
                for y in range(0, self.areaWidth):
                        if self.map[y][column] == 0:
                          #   if math.sqrt(numpy.square(abs(y-Cx))+numpy.square(abs(column-Cy)))>self.r:   #黑洞外
                              emptyLoc.append((y, column))

                  #一列最多的放置节点数
                if len(emptyLoc)<maxNumberInColumn:
                     max=len(emptyLoc)/2
                else:
                    max=maxNumberInColumn

                index = random.randint(0, len(emptyLoc)-1) #产生离散均匀分布的整数，这些整数大于等于low，小于high
                x = emptyLoc[index][1]
                y = emptyLoc[index][0]


                self.map[y][x] = self.map[y][x] + 1

                vertex = Vertex('R'+str(i))
                vertex.setX(x)
                vertex.setY(y)
                self.vertexSet.append(vertex)

                placedNumber = placedNumber + 1
                if placedNumber >= max:
                    column = column + 1#此时此列不可再放置节点
                    placedNumber = 0

            #寻找所有的点中最左边的那些点的X坐标，以便决定S节点的放置位置
            minx = self.areaWidth - 1
            maxx = 0
            for v in self.vertexSet:
                if v.getX() < minx:
                    minx = v.getX()
                if v.getX() > maxx:
                    maxx = v.getX()
#X轴最左和最右的点集合
            leftVertexSet = []
            rightVertexSet = []
            for v in self.vertexSet:
                if v.getX() == minx:
                    leftVertexSet.append(v)
                if v.getX() == maxx:
                    rightVertexSet.append(v)

            index = random.randint(0, len(leftVertexSet)-1)  #随机数
            randomY = leftVertexSet[index].getY()
            #此时minx与randomY即为S节点的放置位置
            vertex = Vertex('S')
            vertex.setX(minx)
            vertex.setY(randomY)
            self.vertexSet.append(vertex)#加入S节点

            index = random.randint(0, len(rightVertexSet)-1)
            randomY = rightVertexSet[index].getY()
            #此时maxx和randomY即为D节点的放置位置
            vertex = Vertex('D')
            vertex.setX(maxx)
            vertex.setY(randomY)
            self.vertexSet.append(vertex)  # 加入D节点

            #首先确认同一个格子里是否存在多于1个点，如果存在，那么直接重新生成图
            '''flag = 1
            for vi in self.vertexSet:
                for vj in self.vertexSet:
                    if vi.getID() != vj.getID() and vi.getX() == vj.getX() and vi.getY() == vj.getY():
                        flag = 0
                        break
                if flag == 0:
                    break
            if flag == 0:
                continue'''

            #点集已经构造完成，接下来构造边集
            for vi in self.vertexSet:
                for vj in self.vertexSet:
                    if vi.getID() != vj.getID():
                        if abs(vi.getX() - vj.getX()) <= 1 and abs(vi.getY() - vj.getY()) <= 1:
                            #说明vi和vj连通
                            edge = Edge(vi, vj)
                            self.edgeSet.append(edge)

            #接下来利用点集和边集构造图，确认图的有效性
            g = Graph(self.vertexSet, self.edgeSet)
            flag = 1
            for vi in self.vertexSet:
                if vi.getID() != 'D':
                    if g.find_path(vi.getID(), 'D', []) is None:
                        flag = 0 #代表生成的图无效，因为有节点不能联通sink
                        break

            #print('flag: '+str(flag))

            #接下来确定路径个数不能过多，如果过多，图无效
            if flag != 0:
                pathSet = g.find_all_paths('S', 'D', [])
                print('生成路径条数检验: '+str(len(pathSet)))
                if len(pathSet) > self.vertexCount * 20 or len(pathSet) < self.vertexCount * 5:
                #if len(pathSet) > self.vertexCount * 10 or len(pathSet) < self.vertexCount :
                    flag = 0 #路径的个数太多或太少，不合适

        #到此，图信息生成结束，可以取出点集和边集了


