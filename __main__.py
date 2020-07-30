from Avg_global import *
from Avg_item import *
from Avg_user import *
from CB_main import *
from CF_main import *
from MIX_result_weigh import *
from MIX_algorithm_fusion import *
from MIX_PROFILE import *

def main():
    flag = int(-1)
    while flag != 0:
        print("--------------------------")
        print("---------平均值法----------")
        print("1.全局平均值")
        print("2.用户对物品评分平均值")
        print("3.物品被用户评分平均值")
        print("-----基于领域的推荐方法-----")
        print("4.基于内容的推荐算法")
        print("5.协同过滤的推荐算法")
        print("--------混合推荐法---------")
        print("6.结果加权式混合推荐算法")
        print("7.算法融合式混合推荐算法")
        print("8.特征组合式混合推荐算法")
        print("0.退出")
        print("--------------------------")
        flag = int(input("输入推荐算法编号:"))
        if flag == 1:
            avg_global()
        elif flag == 2:
            avg_user()
        elif flag == 3:
            avg_item()
        elif flag == 4:
            CB()
        elif flag == 5:
            CF()
        elif flag == 6:
            MIX_R()
        elif flag == 7:
            MIX_A()
        elif flag == 8:
            MIX_P()
        else:
            print("结束")


if __name__ == '__main__':
    main()
