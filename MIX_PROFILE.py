# 特征组合式混合推荐主体
from Package import *


def MIX_P():
    while 1:
        print("--------------------------")
        print("算法融合式混合推荐算法")
        print("1.默认采用余弦相似度")
        print("0.退出")
        print("--------------------------")
        algorithm = int(input("请输入相似度算法选择:"))
        if algorithm == 0:
            break
        else:
            KNN_K = int(input("请输入KNN模型中的K值:"))
            print("--------------------------")
            print("特征组合方式")
            print("1.加")
            print("2.减")
            print("3.乘")
            print("--------------------------")
            flag = int(input("请输入特征组合方式:"))
            if 0 < algorithm < 2 and KNN_K > 0 and 0 < flag < 4:
                Mix_Profile(algorithm, KNN_K, flag)
            else:
                print("参数输入错误，重新输入")


def Mix_Profile(algorithm, KNN_K, flag):
    starttime = datetime.datetime.now()
    print("reading the train and test set....")
    if flag == 1:
        df_test = pd.read_csv("./data/test_add.csv")
        df_train = pd.read_csv("./data/train_add.csv")
    elif flag == 2:
        df_test = pd.read_csv("./data/test_sub.csv")
        df_train = pd.read_csv("./data/train_sub.csv")
    elif flag == 3:
        df_test = pd.read_csv("./data/test_mul.csv")
        df_train = pd.read_csv("./data/train_mul.csv")
    else:
        print("error")
    userId_test = np.array(df_test.iloc[:, 0])
    movieId_test = np.array(df_test.iloc[:, 1])
    rating_test = np.array(df_test.iloc[:, 2])
    profile_test = np.array(df_test.iloc[:, 3:])
    userId_train = np.array(df_train.iloc[:, 0])
    movieId_train = np.array(df_train.iloc[:, 1])
    rating_train = np.array(df_train.iloc[:, 2])
    profile_train = np.array(df_train.iloc[:, 3:])
    left = 0
    right = 0
    rating_pred = []
    while left != len(userId_test):
        while userId_test[right] == userId_test[left]:
            right += 1
            if right == len(userId_test):
                break
        print("left:", left, ", right:", right)
        userId = np.array(df_test.iloc[left:right, 0])
        movieId = np.array(df_test.iloc[left:right, 1])
        test_profile = np.array(df_test.iloc[left:right, 3:])
        train_profile = []
        train_rating = []
        for i in range(0, len(movieId_train)):
            if userId_train[i] == userId[0]:
                train_rating.append(rating_train[i])
                train_profile.append(profile_train[i])
        print(len(movieId))
        print(len(test_profile))
        for i in range(0, len(movieId)):
            recommend_items = Calc_Similarity(test_profile[i], train_profile, algorithm)
            # print(recommend_items)
            a = 0
            b = 0
            score = 0
            if KNN_K <= len(recommend_items):
                for j in range(0, KNN_K):
                    SN = recommend_items[j][0]

                    a += recommend_items[j][1] * train_rating[SN]
                    b += abs(recommend_items[j][1])
            else:
                for j in range(0, len(recommend_items)):
                    SN = recommend_items[j][0]
                    a += recommend_items[j][1] * train_rating[SN]
                    b += abs(recommend_items[j][1])
            if b != 0:
                score = round(a / b, 2)
            else:
                score = 3.5
            rating_pred.append(score)
        left = right

    rating_test = np.array(rating_test)
    rating_pred = np.array(rating_pred)
    print(rating_test)
    print(len(rating_test))
    print(rating_pred)
    print(len(rating_pred))
    MSE = np.sum(np.power((rating_test - rating_pred), 2)) / len(rating_test)
    RMSE = np.sqrt(MSE)
    endtime = datetime.datetime.now()
    print("--------------------------")
    print("共预测", len(rating_pred), "个评分，基于", len(rating_train), "条用户评分数据")
    print("KNN中K的取值为:", KNN_K)
    print("RMSE:", RMSE)
    print("Spend_time:", (endtime - starttime).seconds)
    print("--------------------------")


def Calc_Similarity(profile, profile_list, flag):
    recommend_items = []
    if flag == 1:
        for i in range(0, len(profile_list)):
            Similarity = Cosine(profile, profile_list[i])
            recommend_items.append([i, Similarity])
        recommend_items.sort(key=lambda item: item[1], reverse=True)
    elif flag == 2:
        for i in range(0, len(profile_list)):
            Similarity = Pearson(profile, profile_list[i])
            recommend_items.append([i, Similarity])
        recommend_items.sort(key=lambda item: item[1], reverse=True)
    elif flag == 3:
        for i in range(0, len(profile_list)):
            Similarity = Jaccard(profile, profile_list[i])
            recommend_items.append([i, Similarity])
        recommend_items.sort(key=lambda item: item[1], reverse=True)
    elif flag == 4:
        for i in range(0, len(profile_list)):
            Similarity = Calc_entropy_grap(profile, profile_list[i])
            recommend_items.append([i, Similarity])
        recommend_items.sort(key=lambda item: item[1], reverse=True)
    elif flag == 5:
        for i in range(0, len(profile_list)):
            Similarity = Euclidean(profile, profile_list[i])
            recommend_items.append([i, Similarity])
        recommend_items.sort(key=lambda item: item[1], reverse=False)
        l = len(recommend_items)
        min = recommend_items[0][1]
        max = recommend_items[l - 1][1]
        for j in range(0, l):
            recommend_items[j][1] = 1 - MaxMinCalc(recommend_items[j][1], min, max)
    elif flag == 6:
        for i in range(0, len(profile_list)):
            Similarity = Manhattann(profile, profile_list[i])
            recommend_items.append([i, Similarity])
        recommend_items.sort(key=lambda item: item[1], reverse=False)
        l = len(recommend_items)
        min = recommend_items[0][1]
        max = recommend_items[l - 1][1]
        for j in range(0, l):
            recommend_items[j][1] = 1 - MaxMinCalc(recommend_items[j][1], min, max)
    elif flag == 7:
        for i in range(0, len(profile_list)):
            Similarity = Chebyshevn(profile, profile_list[i])
            recommend_items.append([i, Similarity])
        recommend_items.sort(key=lambda item: item[1], reverse=False)
        l = len(recommend_items)
        min = recommend_items[0][1]
        max = recommend_items[l - 1][1]
        for j in range(0, l):
            recommend_items[j][1] = 1 - MaxMinCalc(recommend_items[j][1], min, max)
    elif flag == 8:
        for i in range(0, len(profile_list)):
            Similarity = Hamming(profile, profile_list[i])
            recommend_items.append([i, Similarity])
        recommend_items.sort(key=lambda item: item[1], reverse=False)
        l = len(recommend_items)
        min = recommend_items[0][1]
        max = recommend_items[l - 1][1]
        for j in range(0, l):
            recommend_items[j][1] = 1 - MaxMinCalc(recommend_items[j][1], min, max)
    else:
        print("error")
        return None
    return recommend_items


if __name__ == '__main__':
    MIX_P()
