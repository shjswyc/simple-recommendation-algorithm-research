# 结果加权式混合推荐主体
from Package import *


def MIX_R():
    while 1:
        print("--------------------------")
        print("结果加权式混合推荐算法")
        print("1.默认采用余弦相似度")
        print("0.退出")
        print("--------------------------")
        algorithm = int(input("请输入相似度算法选择:"))
        if algorithm == 0:
            break
        else:
            KNN_K = int(input("请输入KNN模型中的K值:"))
            weight = float(input("请输入CB的权重值(0~1):"))
            if 0 < algorithm < 2 and KNN_K > 0 and 0 <= weight <= 1:
                Mix_Result_Weigh(algorithm, KNN_K, weight)
            else:
                print("参数输入错误，重新输入")


def Mix_Result_Weigh(algorithm, KNN_K, weight):
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
    print("reading the item and user profile....")
    df_item_profile = pd.read_csv("./data/item_profile.csv")
    item_id = np.array(df_item_profile.iloc[:, 0])
    item_profile = np.array(df_item_profile.iloc[:, 1:])
    df_user_profile = pd.read_csv("./data/user_profile.csv")
    user_id = np.array(df_user_profile.iloc[:, 0])
    user_profile = np.array(df_user_profile.iloc[:, 1:])
    print("content based")
    rating_pred = []
    for i in range(0, len(movieId_test)):
        a = 0
        b = 0
        score1 = 0
        print("i", i)
        print("CB")
        test_item_profile = []
        for j in range(0, len(item_id)):
            if movieId_test[i] == item_id[j]:
                test_item_profile = item_profile[j]
                break
        train_item_profile_list = []
        train_item_rating = []
        for m in range(0, len(movieId_train)):
            if userId_train[m] == userId_test[i]:
                train_item_rating.append(rating_train[m])
                for n in range(0, len(item_id)):
                    if movieId_train[m] == item_id[n]:
                        train_item_profile_list.append(item_profile[n])
                        break
        recommend_items = Calc_Similarity(test_item_profile, train_item_profile_list, algorithm)
        # print(recommend_items)
        if KNN_K <= len(recommend_items):
            for j in range(0, KNN_K):
                SN = recommend_items[j][0]
                a += recommend_items[j][1] * train_item_rating[SN]
                b += abs(recommend_items[j][1])
        else:
            for j in range(0, len(recommend_items)):
                SN = recommend_items[j][0]
                a += recommend_items[j][1] * train_item_rating[SN]
                b += abs(recommend_items[j][1])
        # 计算结果
        if b != 0:
            # print("a:",a,"b:",b)
            score1 = a / b
        else:
            score1 = 3.5
        print("CF")
        a = 0
        b = 0
        score2 = 0
        test_user_profile = []
        for j in range(0, len(user_id)):
            if userId_test[i] == user_id[j]:
                test_user_profile = user_profile[j]
                break
        train_user_profile_list = []
        train_user_rating = []
        for m in range(0, len(movieId_train)):
            if movieId_train[m] == movieId_test[i]:
                train_user_rating.append(rating_train[m])
                for n in range(0, len(user_id)):
                    if userId_train[m] == user_id[n]:
                        train_user_profile_list.append(user_profile[n])
                        break
        recommend_items = Calc_Similarity(test_user_profile, train_user_profile_list, 1)
        # print(recommend_items)
        if KNN_K <= len(recommend_items):
            for j in range(0, KNN_K):
                SN = recommend_items[j][0]
                a += recommend_items[j][1] * train_user_rating[SN]
                b += abs(recommend_items[j][1])
        else:
            for j in range(0, len(recommend_items)):
                SN = recommend_items[j][0]
                a += recommend_items[j][1] * train_user_rating[SN]
                b += abs(recommend_items[j][1])
        # 计算结果
        if b != 0:
            score2 = a / b
        else:
            score2 = 3.5
        #结果加权
        score = round(weight * score1 + (1-weight) * score2, 2)
        rating_pred.append(score)

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
    print("选择的相似度算法为:", algorithm)
    print("KNN中K的取值为:", KNN_K)
    print("CB的权重为:", weight)
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
    MIX_R()
