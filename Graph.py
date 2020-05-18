class Vertex:
    ID = ''
    x = 0  # 所在格子坐标
    y = 0  # 所在格子坐标

    def __init__(self, ID):
        self.ID = ID

    def getID(self):
        return self.ID

    def setX(self, x):
        self.x = x

    def setY(self, y):
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def __repr__(self):
        return 'Vertex(%s)' % repr(self.ID)
        # __repr__返回表达式， __str__返回可阅读信息

    __str__ = __repr__  # 使其指向同一个函数


class Edge(tuple):
    # 继承自建tuple类型并重写new方法
    def __new__(cls, e1, e2):
        return tuple.__new__(cls, (e1, e2))

    def __repr__(self):
        return "Edge(%s, %s)" % (repr(self[0]), repr(self[1]))

    def get_v0ID(self):
        return self[0].getID()

    def get_v1ID(self):
        return self[1].getID()

    __str__ = __repr__


class Graph(dict):
    def __init__(self, vs=[], es=[]):
        """ 建立一个新的图，(vs)为顶点vertices列表，(es)为边缘edges列表 """
        for v in vs:
            self.add_vertex(v)

        for e in es:
            self.add_edge(e)

    def add_vertex(self, v):
        """ 添加顶点 v: 使用字典结构"""
        if v.getID() not in self:
            self[v.getID()] = []

    def add_edge(self, e):
        #添加边缘 e: e 为一个元组(v,w) ,在两个顶点 w 和 v 之间添加成员e ，如果两个顶点之间已有边缘，则替换之
        v0ID = e.get_v0ID()
        v1ID = e.get_v1ID()

        # 由于一条边会产生两个点，因此该实现代表了一个无向图
        if v1ID not in self[v0ID]:
            self[v0ID].append(v1ID)

        if v0ID not in self[v1ID]:
            self[v1ID].append(v0ID)

    #找到一条路径
    def find_path(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if start not in self:
            return None
        for node in self[start]:
            if node not in path:
                newpath = self.find_path(node, end, path)
                if newpath:
                    return newpath
        return None

    #找到所有路径
    def find_all_paths(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if start not in self:
            return []
        paths = []
        for node in self[start]:
            if node not in path:
                newpaths = self.find_all_paths(node, end, path)

                for newpath in newpaths:
                    paths.append(newpath)

        return paths

    #找到最短路径
    def find_shortest_path(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if start not in self:
            return None
        shortest = None
        for node in self[start]:
            if node not in path:
                newpath = self.find_shortest_path(node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest
