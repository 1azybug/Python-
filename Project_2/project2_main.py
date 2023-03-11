import os.path
import tkinter as tk
from crawl import *
import pandas as pd
from tkinter import ttk

if __name__ == '__main__':

    root = tk.Tk()
    root.geometry('1200x400')
    text = tk.Entry()
    text.grid()

    df = pd.DataFrame([])
    city = '沈阳'


    def search_house():
        global df, city

        btn['state'] = tk.DISABLED
        city = text.get()
        if os.path.exists(r'./database/' + city + '.csv'):
            df = pd.read_csv(r'./database/' + city + '.csv')
        else:
            df = crawl_city(city)

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


    btn = tk.Button(root, text='搜索', command=search_house)
    btn.grid(row=0, column=1)


    def save_fun():
        if not os.path.exists(r'./database'):
            os.mkdir(r'./database')
        df.to_csv(r'./database/' + city + ".csv", index=False)
        logger_info = record(filename='./log.txt', level=logging.INFO)
        logger_info.info("爬取信息已保存至" + r'./database/' + city + ".csv")

    save_btn = tk.Button(root, text='保存数据', command=save_fun)
    save_btn.grid(row=0, column=2)
    root.mainloop()
