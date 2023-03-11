import socket
from crawl import *
import pandas as pd
import os.path
import time
import json


def send_all(server, long_str):
    str_len = len(long_str)
    for i in range(str_len // 32 + 1):
        server.send(long_str[i * 32:(i + 1) * 32].encode('utf-8'))


def send_file(sender, file):
    with open(file, 'rb') as f:
        for line in f.readlines():
            sender.send(line)

    time.sleep(1)
    sender.send(b'<finish>')



def send_df(sender, x_df):
    x_df.to_csv('..\\Server\\.Cache', index=False)
    send_file(sender, '..\\Server\\.Cache')


def save_fun():
    global df, city
    if not os.path.exists('..\\Server\\database\\'):
        os.mkdir('..\\Server\\database\\')
    df.to_csv('..\\Server\\database\\' + city + ".csv", index=False)
    logger_info = record(filename='..\\Server\\log.txt', level=logging.INFO)
    logger_info.info("爬取信息已保存至" + '..\\Server\\database\\' + city + ".csv")


def send_num(sender):
    database_path = '..\\Server\\database\\'

    cities = []
    nums = []
    for root, dirs, files in os.walk(database_path):
        for file in files:
            csv_path = os.path.join(root, file)
            tmp_df = pd.read_csv(csv_path)
            cities.append(file.split('.')[0])
            nums.append(tmp_df.shape[0])

    tmp_df = pd.DataFrame({"x": cities, "y": nums})
    tmp_df.to_csv('..\\Server\\.Cache', index=False)
    send_file(sender, '..\\Server\\.Cache')


s = socket.socket()
s.bind(('127.0.0.1', 8298))
s.listen()

df = pd.DataFrame([])
city = '沈阳'

os.startfile(r'..\Server\AutoUpdate.bat')
while True:
    con, addr = s.accept()

    data = (con.recv(1024)).decode('utf-8')
    req, text = data.split('<sep>')
    print(req)
    print('..')
    if req == '<Query>':
        city = text
        if os.path.exists('..\\Server\\database\\' + city + '.csv'):
            df = pd.read_csv('..\\Server\\database\\' + city + '.csv')
        else:
            df = crawl_city(city)
        send_df(con, df)

    if req == '<Save>':
        save_fun()

    if req == '<Show>':
        send_num(con)

    con.close()

