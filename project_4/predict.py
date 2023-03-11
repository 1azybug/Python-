import torch
import pandas as pd
import numpy as np
from data_preprocessing import *


class FNN(torch.nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(70, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, 8),
            torch.nn.Sigmoid(),
            torch.nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layer(x)


def test_loss():
    model = FNN()
    model_dict = torch.load('best.pt')
    model.load_state_dict(model_dict)

    test = pd.read_csv('test.csv')

    batch_size = 1
    X = test.drop(['价格（单位：人民币）'], axis=1).values
    Y = test['价格（单位：人民币）'].values

    k = 0
    mse = torch.nn.MSELoss()
    losses = []
    while k * batch_size < X.shape[0]:
        x = torch.Tensor(X[k * batch_size:(k + 1) * batch_size, :])
        y = torch.Tensor(Y[k * batch_size:(k + 1) * batch_size])
        pred = model(x)
        #         print(k,torch.squeeze(pred,axis=1).shape,y.shape)
        loss = mse(torch.squeeze(pred, axis=1), y)
        losses.append(loss.item())
        k += 1
    print('Test loss=', np.mean(losses))
    return np.mean(losses)


def query():
    Q_dict = {}
    df = pd.read_csv("沈阳.csv")
    for col in df.columns.to_list():
        # print(col, type(col))
        if "Unnamed" in col:
            continue
        if "城市" in col:
            Q_dict[col] = "沈阳"
            continue

        if "链接" in col:
            Q_dict[col] = "http://"
            continue

        if "套内面积" in col:
            Q_dict[col] = "暂无数据"
            continue

        if "挂牌时间" in col:
            Q_dict[col] = "2022-06-29"
            continue

        if "上次交易" in col:
            Q_dict[col] = "2018-07-09"
            continue

        if "价格（单位：人民币）" in col:
            Q_dict[col] = 0.0
            continue

        if "建筑面积" in col:
            area = input("输入建筑面积(单位:㎡),示例:100.1\n")
            Q_dict[col] = area + "㎡"
            continue

        if "户型结构" in col:
            value = input("输入" + col + ",示例:" + "(复式|平层|跃层|错层)" + '\n')
            Q_dict[col] = value
            continue

        if "建筑类型" in col:
            value = input("输入" + col + ",示例:" + "(塔楼|板塔结合|板楼)" + '\n')
            Q_dict[col] = value
            continue

        if "房屋朝向" in col:
            value = input("输入" + col + ",示例:" + "(东|东 东南|东 北 南|东 南 西 北|南 西北|...|南 东南)" + '\n')
            Q_dict[col] = value
            continue

        if "建筑结构" in col:
            value = input("输入" + col + ",示例:" + "(未知结构|框架结构|混合结构|砖混结构|钢混结构)" + '\n')
            Q_dict[col] = value
            continue

        if "装修情况" in col:
            value = input("输入" + col + ",示例:" + "(其他|毛坯|简装|精装)" + '\n')
            Q_dict[col] = value
            continue

        if "供暖方式" in col:
            value = input("输入" + col + ",示例:" + "(自供暖|集中供暖)" + '\n')
            Q_dict[col] = value
            continue

        if "配备电梯" in col:
            value = input("输入" + col + ",示例:" + "(无|有)" + '\n')
            Q_dict[col] = value
            continue

        if "交易权属" in col:
            value = input("输入" + col + ",示例:" + "(动迁安置房|商品房|已购公房|经济适用房)" + '\n')
            Q_dict[col] = value
            continue

        if "房屋用途" in col:
            value = input("输入" + col + ",示例:" + "(别墅|商住两用|普通住宅)" + '\n')
            Q_dict[col] = value
            continue

        if "房屋年限" in col:
            value = input("输入" + col + ",示例:" + "(未满两年|满五年)" + '\n')
            Q_dict[col] = value
            continue

        if "产权所属" in col:
            value = input("输入" + col + ",示例:" + "(共有|非共有)" + '\n')
            Q_dict[col] = value
            continue

        if "房本备件" in col:
            value = input("输入" + col + ",示例:" + "(已上传房本照片|未上传房本照片)" + '\n')
            Q_dict[col] = value
            continue

        if "别墅类型" in col:
            value = input("输入" + col + ",示例:" + "(联排|非别墅)" + '\n')
            Q_dict[col] = value
            continue

        if "所在楼层" in col:
            value = input("输入" + col + ",示例:" + "{中楼层 (共6层)|高楼层 (共11层)|低楼层 (共33层)}" + '\n')
            Q_dict[col] = value
            continue

        if "抵押信息" in col:
            value = input("输入" + col + ",示例:" + "{无抵押|有抵押 30万元|有抵押 80万元}" + '\n')
            Q_dict[col] = value
            continue

        value = input("输入" + col + ",示例:" + df.loc[0, col] + '\n')
        Q_dict[col] = value

    Q_df = pd.DataFrame([Q_dict])
    Q_df.to_csv("Q.csv")


def pred_price():
    model = FNN()
    model_dict = torch.load('best.pt')
    model.load_state_dict(model_dict)

    test = pd.read_csv('cleaned_Q.csv')

    batch_size = 1
    X = test.drop(['价格（单位：人民币）'], axis=1).values
    # print(X)
    k = 0
    while k * batch_size < X.shape[0]:
        #         print(X.shape)
        #         print(X[k*batch_size:(k+1)*batch_size,:].shape)
        x = torch.Tensor(X[k * batch_size:(k + 1) * batch_size, :])
        #         print(x.shape)
        pred = model(x)
        return pred.item()


# inverse_standardize
def inverse_price(price):
    train = pd.read_csv('沈阳.csv')
    scaler = StandardScaler()
    scaler.fit(np.log(train[['价格（单位：人民币）']] + 1))
    #     print(price)
    price = scaler.inverse_transform([[price]])
    #     print(price[0][0])
    return np.exp(price[0][0]) - 1


def cosin_sim(x, y):
    return np.sum(x * y) / (np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)))


def get_k_top_sim(k=5):
    raw_df = pd.read_csv("沈阳.csv")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    df = pd.concat([train_df, test_df], axis=0)
    df = df.reset_index(drop=True)
    base_df = pd.read_csv("cleaned_Q.csv")
    base_df.loc[0, "价格（单位：人民币）"] = pred_price()
    #     print(df.shape,raw_df.shape,base_df.shape)

    sim_list = []
    for i in range(df.shape[0]):
        #         print(np.array(df.iloc[i,:]).shape,np.array(base_df.iloc[0,:]).shape)
        x = np.array(df.iloc[i, :])
        y = np.array(base_df.iloc[0, :])
        sim_list.append(cosin_sim(x, y))

    raw_df["余弦相似度"] = sim_list

    raw_df = raw_df.sort_values(by="余弦相似度", axis=0, ascending=False)
    return raw_df.iloc[:k, :]


if __name__ == "__main__":
    # query()
    # test_process()
    # inverse_price(pred_price())
    test_loss()
