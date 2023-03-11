import tkinter as tk
import pandas as pd
from tkinter import ttk
from predict import *


def show_k_top(price, k=5):
    root = tk.Tk()
    root.geometry('1200x400')

    lable = tk.Label(text="预测价格为:", )

    df = get_k_top_sim(k).drop(columns=['Unnamed: 0'])

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

    root.mainloop()


if __name__ == "__main__":
    query()
    test_process()
    show_k_top(inverse_price(pred_price()))
