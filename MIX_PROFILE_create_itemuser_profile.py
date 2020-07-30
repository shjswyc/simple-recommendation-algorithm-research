# 构建物品和用户特征向量的组合
from Package import *


def create_itemuser_profile():
    # df_rating = pd.read_csv("./data/test.csv")
    df_rating = pd.read_csv("./data/train.csv")
    user_list = np.array(df_rating.iloc[:, 1])
    movie_list = np.array(df_rating.iloc[:, 2])
    rank_list = np.array(df_rating.iloc[:, 3])
    df_item_profile = pd.read_csv("./data/item_profile.csv")
    movie_id = np.array(df_item_profile.iloc[:, 0])
    movie_profile = np.array(df_item_profile.iloc[:, 1:])
    df_user_profile = pd.read_csv("./data/user_profile.csv")
    user_id = np.array(df_user_profile.iloc[:, 0])
    user_profile = np.array(df_user_profile.iloc[:, 1:])
    # print(movie_id)
    result = list()
    for i in range(0, len(user_list)):
        tmp1 = []
        tmp2 = []
        for m in range(0, len(user_id)):
            if user_list[i] == user_id[m]:
                tmp1 = np.array(user_profile[m])
                break
        for n in range(0, len(movie_id)):
            if movie_list[i] == movie_id[n]:
                tmp2 = np.array(movie_profile[n])
                break
        tmp = np.hstack((user_list[i], movie_list[i]))
        tmp = np.hstack((tmp, rank_list[i]))
        tmp3 = tmp2 * tmp1
        tmp3 = MaxMinNormalization(tmp3)
        tmp = np.hstack((tmp, tmp3))
        result.append(tmp)

    print(result)
    data = pd.DataFrame(result)
    data.to_csv('./data/train_mul.csv',index=False)


if __name__ == '__main__':
    create_itemuser_profile()
