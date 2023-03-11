import time
import pyautogui as ui
import os
import pyperclip
from tqdm import tqdm

ui.FAILSAFE = False
ui.PAUSE = 1

run_server_cmd = r'python ..\Server\Server.py'
run_client_cmd = r'python Client.py'


# async def run_server():
#     os.system(run_server_cmd)
#
#
# async def run_client():
#     os.system(run_client_cmd)

os.startfile(r'..\Server\run.bat')
os.startfile(r'..\Client\run.bat')


time.sleep(10)


capitals = {'湖南': '长沙', '湖北': '武汉', '广东': '广州', '广西': '南宁', '河北': '石家庄', '河南': '郑州', '山东': '济南',
            '山西': '太原', '江苏': '南京', '浙江': '杭州', '江西': '南昌', '黑龙江': '哈尔滨', '新疆': '乌鲁木齐', '云南': '昆明',
            '贵州': '贵阳', '福建': '福州', '吉林': '长春', '安徽': '合肥', '四川': '成都', '西藏': '拉萨', '宁夏': '银川',
            '辽宁': '沈阳', '青海': '西宁', '甘肃': '兰州', '陕西': '太原', '内蒙古': '呼和浩特', '台湾': '台北', '北京': '北京',
            '上海': '上海', '天津': '天津', '重庆': '重庆', '香港': '香港', '澳门': '澳门'}


ui.click(600, 185, button='left')
time.sleep(3)
ui.hotkey('alt', 'f4')
cities = list(capitals.values())
for i in tqdm(range(len(capitals))):
    city = cities[i]
    ui.doubleClick(300, 185, button='left')
    pyperclip.copy(city)
    ui.hotkey('ctrl', 'v')
    ui.click(465, 185, button='left')
    time.sleep(3)
    ui.click(520, 185, button='left')
    time.sleep(3)
