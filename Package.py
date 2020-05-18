class Package:
    ID = ""
    flag = "" #'T'代表被篡改 'D'代表丢弃 'R'代表重复 'N'代表正常

    def __init__(self, ID, flag):
        self.ID = ID
        self.flag = flag

    def getID(self):
        return self.ID

    def getFlag(self):
        return self.flag

    def setFlag(self, flag):
        self.flag = flag
