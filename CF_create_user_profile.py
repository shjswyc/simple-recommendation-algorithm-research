# 构建用户兴趣的特征向量
from Package import *


def create_user_profile():
    print("reading the source ratings messages....")
    df_rating = pd.read_csv("./source/ratings.csv")
    user_list = np.array(df_rating.iloc[:, 0])
    df_profile = pd.read_csv("./data/item_profile.csv")
    movie_id = np.array(df_profile.iloc[:, 0])
    movie_profile = np.array(df_profile.iloc[:, 1:])
    left = 0
    right = 0
    profile_list = list()
    while left != len(user_list):
        while user_list[right] == user_list[left]:
            right += 1
            if right == len(user_list):
                break
        print("left:", left, ", right:", right)
        # 对一个用户求profile
        userId = np.array(df_rating.iloc[left:right, 0])
        movieId = np.array(df_rating.iloc[left:right, 1])
        rating = np.array(df_rating.iloc[left:right, 2])
        # rating_avg = np.average(rating)
        # print(rating_avg)
        profile = [0] * 19
        for i in range(0, len(movieId)):
            # 关联两张表的过程
            for j in range(0, len(movie_id)):
                if movieId[i] == movie_id[j]:
                    # profile += (rating[i] - rating_avg) * movie_profile[j]
                    profile += (rating[i] - 2.75) * movie_profile[j]
        profile = MaxMinNormalization(profile)
        profile_list.append(profile)
        left = right
    result = pd.DataFrame(profile_list)
    user = []
    for user_count in range(1,611):
        user.append(user_count)
    result.insert(0, 'userId', user)
    result.to_csv("./data/user_profile.csv", index=False)


if __name__ == '__main__':
    create_user_profile()
