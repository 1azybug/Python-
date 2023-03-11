import sklearn
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def get_train_test():
    df = pd.read_csv('沈阳.csv')

    # delete useless columns and mostly NAN columns
    df['别墅类型'].fillna("非别墅", inplace=True)
    df = df.drop(columns=['Unnamed: 0', '城市', '链接', '挂牌时间', '上次交易'])

    # transform data
    df_dict = df.to_dict()

    # df_dict['房屋户型']
    shi_dict = {}
    ting_dict = {}
    chu_dict = {}
    wei_dict = {}
    for index, num in df_dict['房屋户型'].items():
        shi_dict[index] = float(num.split('室')[0][-1])
        ting_dict[index] = float(num.split('厅')[0][-1])
        chu_dict[index] = float(num.split('厨')[0][-1])
        wei_dict[index] = float(num.split('卫')[0][-1])
    #     print(index,num,num.split('室')[0][-1],num.split('厅')[0][-1],num.split('厨')[0][-1],num.split('卫')[0][-1])
    df_dict['室'] = shi_dict
    df_dict['厅'] = ting_dict
    df_dict['厨'] = chu_dict
    df_dict['卫'] = wei_dict

    # df_dict['所在楼层']
    floor_dict = {}
    num_floor_dict = {}
    for index, info in df_dict['所在楼层'].items():
        floor_dict[index] = info.split('(')[0]
        num_floor_dict[index] = float(info.split('(共')[1].split('层)')[0])
    df_dict['所在层'] = floor_dict
    df_dict['总层数'] = num_floor_dict

    # df_dict['建筑面积']
    for index, info in df_dict['建筑面积'].items():
        df_dict['建筑面积'][index] = float(info.split('㎡')[0])

    # df_dict['套内面积']
    for index, info in df_dict['套内面积'].items():
        #     print(info)
        if info == '暂无数据':
            df_dict['套内面积'][index] = np.nan
            continue
        df_dict['套内面积'][index] = float(info.split('㎡')[0])

    # df_dict['抵押信息']
    for index, info in df_dict['抵押信息'].items():
        #     print(info)
        if info.strip() == '无抵押':
            df_dict['抵押信息'][index] = float(0.0)
            continue

        if '万元' not in info:
            df_dict['抵押信息'][index] = np.nan
            continue

        #     print([info])
        df_dict['抵押信息'][index] = float(info.split('抵押 ')[1].split('万元')[0]) * 10000.0

    cn_num = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
              "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15, "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
              "两": 2, "二十一": 21, "三十二": 32
              }

    # df_dict['梯户比例']
    ti_dict = {}
    hu_dict = {}
    for index, info in df_dict['梯户比例'].items():
        #     print(info)

        if info is np.nan:
            continue

        ti_dict[index] = float(cn_num[info.split('梯')[0]])
        hu_dict[index] = float(cn_num[info.split('梯')[1].split('户')[0]])
    #     print(ti_dict[index],hu_dict[index])
    df_dict['梯数'] = ti_dict
    df_dict['户数'] = hu_dict

    df = pd.DataFrame(df_dict)

    df = df.drop(columns=['房屋户型', '所在楼层', '梯户比例', '套内面积'])
    df = df.replace('暂无数据', np.nan)

    # record for standardize
    num_cols = [i for i in df.columns if df[i].dtype in ['int64', 'float64']]

    # one-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Fill missing value with KNN
    imp = KNNImputer(n_neighbors=10, weights="uniform")
    df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)

    # standardize
    df[num_cols] = np.log(df[num_cols] + 1)

    scaler = StandardScaler()

    df_normalized = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

    for col in num_cols:
        df[col] = df_normalized[col]

    # train_test_split 8:2 and save
    house_num = df.shape[0]
    train_prepared = df.iloc[:int(house_num * 0.8), :]
    test_prepared = df.iloc[int(house_num * 0.8):, :]
    train_prepared.to_csv('train.csv', index=False)
    test_prepared.to_csv('test.csv', index=False)


def test_process(csv_name='Q.csv'):
    train = pd.read_csv('沈阳.csv')
    test = pd.read_csv(csv_name)
    df = pd.concat([train, test], axis=0)

    # delete useless columns and mostly NAN columns
    df['别墅类型'].fillna("非别墅", inplace=True)
    df = df.drop(columns=['Unnamed: 0', '城市', '链接', '挂牌时间', '上次交易'])

    # print(df)
    # transform data
    df_dict = df.to_dict()

    # df_dict['房屋户型']
    shi_dict = {}
    ting_dict = {}
    chu_dict = {}
    wei_dict = {}
    for index, num in df_dict['房屋户型'].items():
        shi_dict[index] = float(num.split('室')[0][-1])
        ting_dict[index] = float(num.split('厅')[0][-1])
        chu_dict[index] = float(num.split('厨')[0][-1])
        wei_dict[index] = float(num.split('卫')[0][-1])
    #     print(index,num,num.split('室')[0][-1],num.split('厅')[0][-1],num.split('厨')[0][-1],num.split('卫')[0][-1])
    df_dict['室'] = shi_dict
    df_dict['厅'] = ting_dict
    df_dict['厨'] = chu_dict
    df_dict['卫'] = wei_dict

    # df_dict['所在楼层']
    floor_dict = {}
    num_floor_dict = {}
    for index, info in df_dict['所在楼层'].items():
        floor_dict[index] = info.split('(')[0]
        num_floor_dict[index] = float(info.split('(共')[1].split('层)')[0])
    df_dict['所在层'] = floor_dict
    df_dict['总层数'] = num_floor_dict

    # df_dict['建筑面积']
    for index, info in df_dict['建筑面积'].items():
        df_dict['建筑面积'][index] = float(info.split('㎡')[0])

    # df_dict['套内面积']
    for index, info in df_dict['套内面积'].items():
        #     print(info)
        if info == '暂无数据':
            df_dict['套内面积'][index] = np.nan
            continue
        df_dict['套内面积'][index] = float(info.split('㎡')[0])

    # df_dict['抵押信息']
    for index, info in df_dict['抵押信息'].items():
        #     print(info)
        if info.strip() == '无抵押':
            df_dict['抵押信息'][index] = float(0.0)
            continue

        if '万元' not in info:
            df_dict['抵押信息'][index] = np.nan
            continue

        #     print([info])
        df_dict['抵押信息'][index] = float(info.split('抵押 ')[1].split('万元')[0]) * 10000.0

    cn_num = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
              "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15, "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
              "两": 2, "二十一": 21, "三十二": 32
              }

    # df_dict['梯户比例']
    ti_dict = {}
    hu_dict = {}
    for index, info in df_dict['梯户比例'].items():
        #     print(info)

        if info is np.nan:
            continue

        ti_dict[index] = float(cn_num[info.split('梯')[0]])
        hu_dict[index] = float(cn_num[info.split('梯')[1].split('户')[0]])
    #     print(ti_dict[index],hu_dict[index])
    df_dict['梯数'] = ti_dict
    df_dict['户数'] = hu_dict

    df = pd.DataFrame(df_dict)

    # print(df)

    df = df.drop(columns=['房屋户型', '所在楼层', '梯户比例', '套内面积'])
    df = df.replace('暂无数据', np.nan)

    # record for standardize
    num_cols = [i for i in df.columns if df[i].dtype in ['int64', 'float64']]

    # one-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Fill missing value with KNN
    imp = KNNImputer(n_neighbors=10, weights="uniform")
    df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)

    # print(df)

    # standardize
    df[num_cols] = np.log(df[num_cols] + 1)

    scaler = StandardScaler()

    df_normalized = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

    for col in num_cols:
        df[col] = df_normalized[col]

    df = df.iloc[-1:, :]
    df.to_csv("cleaned_"+csv_name, index=False)


if __name__ == "__main__":
    test_process(csv_name='Q.csv')
