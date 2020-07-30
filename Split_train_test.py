# 对原始数据集分割训练集、测试集
from Package import *


def split():
    print("reading the source ratings messages....")
    df_rating = pd.read_csv("./source/ratings.csv")
    user_movieId = np.array(df_rating.iloc[:, 0:2])
    rating = np.array(df_rating.iloc[:, 2])
    x_train, x_test, y_train, y_test = train_test_split(user_movieId, rating, test_size=0.01, random_state=1)
    train = list()
    test = list()
    for i in range(0, len(x_train)):
        train.append(np.hstack((x_train[i], y_train[i])))
    print(train)
    df_train = pd.DataFrame(train, columns=["userId", "movieId", "rating"])
    df_train.sort_values(by=["userId", "movieId"], ascending=[True, True], inplace=True)
    df_train.reset_index(drop=True, inplace=True)
    df_train.to_csv('./data/train.csv')
    for i in range(0, len(x_test)):
        test.append(np.hstack((x_test[i], y_test[i])))
    print(test)
    df_test = pd.DataFrame(test, columns=["userId", "movieId", "rating"])
    df_test.sort_values(by=["userId", "movieId"], ascending=[True, True], inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_test.to_csv('./data/test.csv')


if __name__ == '__main__':
    split()
