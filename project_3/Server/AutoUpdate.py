import os
from crawl import *
import datetime
import time

while True:

    last_time = os.path.getmtime(r'..\Server\database\郑州.csv')
    last_time = time.gmtime(last_time)
    last_time = datetime.datetime(*last_time[:6])

    #  crawl once a week
    if (datetime.datetime.now()-last_time).days >= 7:
        print(datetime.datetime.now())
        print(last_time)
        crawl()
    else:
        print("*"*20)
        print("当前时间:", datetime.datetime.now())
        print("上一次修改时间:", last_time)
        print("还不需要更新")
        print("*" * 20)
        time.sleep(5)


