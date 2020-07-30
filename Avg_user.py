# 预测函数可以定义为用户u在训练集中所有评分的平均值
from Package import *


def avg_user():
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
    left = 0
    right = 0
    KNN_K = 5
    rating_pred = []
    while left != len(userId_test):
        while userId_test[right] == userId_test[left]:
            right += 1
            if right == len(userId_test):
                break
        print("left:", left, ", right:", right)
        userId = np.array(df_test.iloc[left:right, 1])
        movieId = np.array(df_test.iloc[left:right, 2])
        score = 0
        count = 0
        for i in range(0, len(movieId_train)):
            if userId_train[i] == userId[0]:
                score += rating_train[i]
                count += 1
        score = score / count
        for i in range(0, len(movieId)):
            rating_pred.append(score)
        left = right

    rating_test = np.array(rating_test)
    rating_pred = np.array(rating_pred)
    print(rating_test)
    print(len(rating_test))
    print(rating_pred)
    print(len(rating_pred))
    RMSE = calc_rmse(rating_test, rating_pred)
    endtime = datetime.datetime.now()
    print("--------------------------")
    print("共预测", len(rating_pred), "个评分，基于", len(rating_train), "条用户评分数据")
    print("KNN中K的取值为:", KNN_K)
    print("RMSE:", RMSE)
    print("Spend_time:", (endtime - starttime).seconds)
    print("--------------------------")


if __name__ == '__main__':
    avg_user()
