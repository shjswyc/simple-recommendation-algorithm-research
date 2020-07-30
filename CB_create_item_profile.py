# 构建物品标签的特征向量
from Package import *


def create_item_profile():
    # 读源数据
    df = pd.read_csv("./source/movies.csv")
    movieId = np.array(df.iloc[:, 0])
    genres = np.array(df.iloc[:, 2])
    label_dict = create_label_set(genres)
    profile = label_to_profile(label_dict, genres)
    profile.insert(0, 'movieId', movieId)
    profile.to_csv("./data/item_profile.csv", index=False)


def create_label_set(genres):
    # 创建集合，利用集合的无序不可重复性来得到所有标签。后根据集合创建字典
    label_set = set()
    for i in range(0, len(genres)):
        tmp = genres[i].split('|')
        label_set = label_set.union(set(tmp))
    label_set.remove('(no genres listed)')
    label_set = list(label_set)
    label_set.sort()
    print("Set:", label_set)
    label_dict = dict()
    value = 0
    for label in label_set:
        label_dict[label] = value
        value += 1
    # Dictionary: {'Action': 0, 'Adventure': 1, 'Animation': 2, 'Children': 3, 'Comedy': 4, 'Crime': 5, 'Documentary': 6,
    #              'Drama': 7, 'Fantasy': 8, 'Film-Noir': 9, 'Horror': 10, 'IMAX': 11, 'Musical': 12, 'Mystery': 13,
    #              'Romance': 14, 'Sci-Fi': 15, 'Thriller': 16, 'War': 17, 'Western': 18}
    print("Dictionary:", label_dict)
    return label_dict


def label_to_profile(label_dict, genres):
    # 构建特征向量
    profile_list = list()
    for i in range(0, len(genres)):
        profile = [0] * len(label_dict)
        tmp = genres[i].split('|')
        for j in range(0, len(tmp)):
            if tmp[j] != '(no genres listed)':
                profile[label_dict[tmp[j]]] = 1
        print(profile)
        profile_list.append(profile)
    result = pd.DataFrame(profile_list)
    return result


if __name__ == '__main__':
    create_item_profile()
