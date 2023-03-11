import socket
import os
import tkinter as tk
import pandas as pd
from tkinter import ttk
import time
import matplotlib.pyplot as plt

def get_df_from_server(one_city):
    client = socket.socket()
    client.connect(('127.0.0.1', 8298))
    client.send(b"<Query>" + b"<sep>" + one_city.encode('utf-8'))

    if os.path.exists('.Cache'):
        os.remove('.Cache')

    with open('.Cache', 'wb') as f:
        f.write(b"")

    while True:
        data = client.recv(1024)
        if data == b"<finish>":
            break
        with open('.Cache', 'ab') as f:
            f.write(data)

    client.close()

    return pd.read_csv('.Cache')





def search_house():
    global df, city

    btn['state'] = tk.DISABLED
    city = text.get()

    df = get_df_from_server(city)

    frame = tk.Frame(root)
    frame.place(x=0, y=50, width=1200, height=300, )

    ybar = ttk.Scrollbar(frame)
    ybar.pack(side='right', fill='y')

    xbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
    xbar.pack(side='bottom', fill='x')

    tree = ttk.Treeview(frame, columns=tuple(df.columns), show='headings',
                        yscrollcommand=ybar.set, xscrollcommand=xbar.set)

    ybar.config(command=tree.yview)
    xbar.config(command=tree.xview)

    for col in df.columns:
        tree.column(col, width=100, anchor='center')
        tree.heading(col, text=col)

    tree.tag_configure("tgOddRow", background="white")
    tree.tag_configure("tgEvenRow", background="lightblue")

    for i in range(df.shape[0]):
        if i % 2 == 0:
            tree.insert("", i, values=tuple(df.iloc[i, :].values), tag="tgOddRow")

        else:
            tree.insert("", i, values=tuple(df.iloc[i, :].values), tag="tgEvenRow")

    tree.pack(side='left', fill='y')
    btn['state'] = tk.NORMAL


def save_fun():
    client = socket.socket()
    client.connect(('127.0.0.1', 8298))
    client.send(b"<Save>" + b"<sep>" + b"<Empty>")
    client.close()

def show_num_houses():
    client = socket.socket()
    client.connect(('127.0.0.1', 8298))
    client.send(b"<Show>" + b"<sep>" + b"<Empty>")

    if os.path.exists('.Cache'):
        os.remove('.Cache')

    with open('.Cache', 'wb') as f:
        f.write(b"")

    while True:
        data = client.recv(1024)
        if data == b"<finish>":
            break
        with open('.Cache', 'ab') as f:
            f.write(data)

    client.close()
    num_data = pd.read_csv('.Cache')

    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111)
    ax.bar(num_data['x'], num_data['y'])
    plt.show()





root = tk.Tk()
root.geometry('1200x400+200+100')
text = tk.Entry()
text.grid()

df = pd.DataFrame([])
city = '沈阳'

btn = tk.Button(root, text='搜索', command=search_house)
btn.grid(row=0, column=1)


save_btn = tk.Button(root, text='保存数据', command=save_fun)
save_btn.grid(row=0, column=2)

save_btn = tk.Button(root, text='二手房数量', command=show_num_houses)
save_btn.grid(row=0, column=3)
root.mainloop()
