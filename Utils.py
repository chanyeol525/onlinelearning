#工具类
class Utils:

    @staticmethod
    def calSimilarity(src, des):
        count = 0

        for element in src:
            if element in des:
                count = count + 1

        return count

