# 它的定义为训练集中所有评分记录的评分平均值。
from Package import *


def avg_global():
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
    sum = 0
    avg = 0
    rating_pred = []
    for i in range(0, len(rating_train)):
        sum += rating_train[i]
        avg = sum / len(rating_train)
    for i in range(0, len(rating_test)):
        rating_pred.append(avg)

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
    print("RMSE:", RMSE)
    print("Spend_time:", (endtime - starttime).seconds)
    print("--------------------------")


if __name__ == '__main__':
    avg_global()
