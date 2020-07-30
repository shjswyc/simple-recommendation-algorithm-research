# 基于内容的推荐算法主体
from Package import *


def CB():
    while 1:
        print("--------------------------")
        print("基于内容的推荐算法")
        print("1.余弦相似度")
        print("2.皮尔逊相关系数")
        print("3.杰卡德系数")
        print("4.信息增益")
        print("5.欧氏距离")
        print("6.曼哈顿距离")
        print("7.切比雪夫距离")
        print("8.汉明距离")
        print("0.退出")
        print("--------------------------")
        algorithm = int(input("请输入相似度算法选择:"))
        if algorithm == 0:
            break
        else:
            KNN_K = int(input("请输入KNN模型中的K值:"))
            if 0 < algorithm < 9 and KNN_K > 0:
                ContentBased(algorithm, KNN_K)
            else:
                print("参数输入错误，重新输入")


def ContentBased(algorithm, KNN_K):
    starttime = datetime.datetime.now()
    print("reading the train and test set....")
    df_test = pd.read_csv("./data/test.csv")
    userId_test = np.array(df_test.iloc[:, 1])
    movieId_test = np.array(df_test.iloc[:, 2])
    rating_test = np.array(df_test.iloc[:, 3])
    df_train = pd.read_csv("./data/train.csv")
    userId_train = np.array(df_train.iloc[:, 1])
    movieId_train = np.array(df_train.iloc[:, 2])
    rating_train = np.array(df_train.iloc[:, 3])
    df_profile = pd.read_csv("./data/item_profile.csv")
    id = np.array(df_profile.iloc[:, 0])
    profile = np.array(df_profile.iloc[:, 1:])
    left = 0
    right = 0
    rating_pred = []
    while left != len(userId_test):
        while userId_test[right] == userId_test[left]:
            right += 1
            if right == len(userId_test):
                break
        print("left:", left, ", right:", right)
        userId = np.array(df_test.iloc[left:right, 1])
        print(userId)
        movieId = np.array(df_test.iloc[left:right, 2])
        train_profile = []
        train_rating = []
        for i in range(0, len(movieId_train)):
            if userId_train[i] == userId[0]:
                train_rating.append(rating_train[i])
                for j in range(0, len(id)):
                    if movieId_train[i] == id[j]:
                        train_profile.append(profile[j])
                        break
        test_profile = []
        for i in range(0, len(movieId)):
            for j in range(0, len(id)):
                if movieId[i] == id[j]:
                    test_profile.append(profile[j])
                    break
        for i in range(0, len(movieId)):
            recommend_items = Calc_Similarity(test_profile[i], train_profile, algorithm)
            # print(recommend_items)
            a = 0
            b = 0
            score = 0
            #KNN
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
    print("test:", rating_test)
    print("pred:", rating_pred)
    RMSE = calc_rmse(rating_test, rating_pred)
    endtime = datetime.datetime.now()
    print("--------------------------")
    print("共预测", len(rating_pred), "个评分，基于", len(rating_train), "条用户评分数据-")
    print("选择的相似度算法为:", algorithm)
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
    CB()
