import numpy as np

from Graph import Vertex, Edge, Graph
from Node import Node
from Package import Package
import random
import math
from GraphInfoMaker import GraphInfoMaker
from Utils import Utils

#建图start
class Network:
    pathset=[]


    def generate(self):
         GI = GraphInfoMaker()
         GI.generate()
         print('节点个数:')
         print(len(GI.getVertexSet()))

         print('边个数:')
         print(len(GI.getEdgeSet()))
         VS = GI.getVertexSet()
         ES = GI.getEdgeSet() #print(VS)
         g = Graph(VS, ES)
         pathSet = g.find_all_paths('S', 'D', [])
         print('地图路径条数为:')
         print(len(pathSet))
#建图end
#路径的逆置
         print("路径的逆置")
         pathRe = pathSet
         for path in pathSet:
             for i in range(0, int(len(path)/2)):
                   node = path[i]
                   path[i] = path[len(path) - 1-i]
                   path[len(path) - 1 - i] = node
         print(pathSet)

#根据百分比去除一些路径start
         useRate = 1
         reduceRate = 1 - useRate
         reduceSet = random.sample(range(0, len(pathSet)), int(len(pathSet)*reduceRate))
         newPathSet = []
         for i in range(0, len(pathSet)):
             if i not in reduceSet:
                 newPathSet.append(pathSet[i])
         print('经过修改多样性后的地图路径条数为(使用率 '+str(useRate)+' ):')
         #pathSet = newPathSet
         return newPathSet,GI
#print(len(pathSet))
#根据百分比去除一些路径end
