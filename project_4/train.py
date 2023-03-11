import torch
import pandas as pd
import numpy as np
from predict import test_loss

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


min_mse = 0.3


def find_min_mse():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    train = train.sample(frac=1)  # 打乱顺序
    val, tr = train.iloc[:int(train.shape[0] * 0.1), :], train.iloc[int(train.shape[0] * 0.1):, :]

    X_train, X_valid = tr.drop('价格（单位：人民币）', axis=1), val.drop('价格（单位：人民币）', axis=1)
    Y_train, Y_valid = tr['价格（单位：人民币）'], val['价格（单位：人民币）']

    X_train = X_train.values
    X_valid = X_valid.values
    Y_train = Y_train.values
    Y_valid = Y_valid.values

    model = FNN()

    def fit():
        epochs = 500
        batch_size = 32

        opt = torch.optim.Adam(model.parameters(), lr=0.0001)
        mse = torch.nn.MSELoss()

        for epoch in range(epochs):
            print(f'epoch:{epoch}')
            k = 0
            while k * batch_size <= X_train.shape[0]:
                opt.zero_grad()
                x = torch.Tensor(X_train[k * batch_size:(k + 1) * batch_size, :])
                y = torch.Tensor(Y_train[k * batch_size:(k + 1) * batch_size])
                pred = model(x)
                loss = mse(torch.squeeze(pred), y)
                if k % 50 == 0:
                    print(f'loss={loss.item()}')
                loss.backward()
                opt.step()
                k += 1

            k = 0
            losses = []
            while k * batch_size <= X_valid.shape[0]:
                x = torch.Tensor(X_valid[k * batch_size:(k + 1) * batch_size, :])
                y = torch.Tensor(Y_valid[k * batch_size:(k + 1) * batch_size])
                pred = model(x)
                loss = mse(torch.squeeze(pred), y)
                losses.append(loss.item())
                k += 1
            print('Valid: loss=', np.mean(losses))

            global min_mse

            if np.mean(losses) < min_mse:
                min_mse = np.mean(losses)
                print("Save to best.pt")
                torch.save(model.state_dict(), 'best.pt')
                with open('min_mse.txt', 'a') as f:
                    f.write(str(min_mse) + '\n')

    fit()


for _ in range(50):
    with open('min_mse.txt', 'a') as f:
        f.write('\nepochs:' + str(_) + '\n')
    find_min_mse()

test_loss()
