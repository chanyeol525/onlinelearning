class Node:
    ID = '0'
    pTA = 0.0  #篡改攻击的概率
    pDA = 0.0  #丢包攻击的概率
    pRA = 0.0  #重放攻击的概率

    def __init__(self, ID, pTA, pDA, pRA):
        self.ID = ID
        self.pTA = pTA
        self.pDA = pDA
        self.pRA = pRA

    def getID(self):
        return self.ID

    def get_pTA(self):
        return self.pTA

    def get_pDA(self):
        return self.pDA

    def get_pRA(self):
        return self.pRA



