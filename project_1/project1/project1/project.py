from xpinyin import Pinyin
import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm
import os
import time
import logging
from log import record
logger_info = record(filename='./log.txt', level=logging.INFO)
path = "china_city_list.csv"
src_url = 'https://github.com/brightgems/china_city_dataset/blob/master/china_city_list.csv'
if os.path.exists(path):
    print("中国城市数据集加载中...")
    logger_info.info('加载数据集')
    for i in tqdm([0]):
        df = pd.read_csv(path, encoding='gbk')
        # time.sleep(1.5)  # 防止加载太快,使进度条错位

    print("中国城市数据集加载完成")
else:
    print("中国城市数据集缺失,请前往 " + src_url + " 下载,置于该脚本同级目录下。")
    logger_info.info("中国城市数据集缺失")
    exit(0)

# print(df.head())
# exit(0)

# 模糊匹配度
# print(fuzz.partial_ratio("this is a test", "this is a test!"))

print(r"支持模糊匹配,在城市名后面添加':f',f为模糊匹配阈值,模糊匹配度超过该阈值的城市会被匹配 ")
print(r"如输入  沈阳:0.5  ,则模糊匹配度超过0.5的城市将会被输出")
pin = Pinyin()


def city_pinyin(str_city):
    list_city = list(str_city)
    str_pinyin = ''
    for py_city in list_city:
        str_pinyin += pin.get_pinyin(py_city, "")[0]
    return str_pinyin


while True:
    city = input("请输入城市名称：")
    city = city.replace('：', ':')
    logger_info.info('搜索城市:' + city)

    th = '1'  # 匹配阈值,1表示完全匹配
    if ':' in city:
        city, th = city.split(':')

    logger_info.info('搜索城市:' + city + ' 阈值:' + th)
    th = float(th)

    cities = []
    for i in tqdm(range(df.shape[0])):
        abbr = df.iloc[i, 0]
        full_name = df.iloc[i, 1]
        abbr_score = fuzz.partial_ratio(abbr, city)
        full_name_score = fuzz.partial_ratio(full_name, city)
        score = max(abbr_score, full_name_score)
        if score >= th * 100:
            cities.append([abbr, score])
            # print('匹配到:' + abbr, '模糊匹配度为:' + str(score) + r'%', "拼音为:" + city_pinyin(abbr))

    cities = sorted(cities, key=lambda x: x[1], reverse=True)

    if cities:
        for c in cities:
            print('匹配到:' + c[0], '模糊匹配度为:' + str(c[1]) + r'%', "拼音为:" + city_pinyin(c[0]))
    else:
        print("无匹配城市")
    logger_info.info('完成一次搜索')
    exit(0)
