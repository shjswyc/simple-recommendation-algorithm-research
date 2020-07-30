# 从表格数据中生成折线图
from Package import *


def create_plt():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    df = pd.read_excel("./data/result.xlsx")
    print(df)
    print(df.columns)
    x = np.array(df.iloc[0:10, 0])
    y = np.array(df.iloc[0:10, 1:])
    print(y)
    plt.plot(x, y)
    plt.legend([df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5],df.columns[6],df.columns[7],df.columns[8]],loc=1)
    # plt.legend([df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5]],loc=1)
    my_x_ticks = np.arange(0, 22, 2)
    my_y_ticks = np.arange(0.9, 1.2, 0.05)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.grid(True)
    plt.xlabel('K值')
    plt.ylabel('RMSE值')
    plt.title('CB与CF下RMSE值对比')
    plt.show()

if __name__ == '__main__':
    create_plt()