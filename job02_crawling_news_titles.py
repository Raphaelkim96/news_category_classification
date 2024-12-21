from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions

from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
import re
import time
import datetime

options = ChromeOptions()
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
options.add_argument('user_agent=' + user_agent)
options.add_argument('lang=ko_KR')

service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

category = ['Politics']
df_titles = pd.DataFrame()
url = 'https://news.naver.com/section/100'
driver.get(url)
button_xpath = '//*[@id="newsct"]/div[4]/div/div[2]' #id가 newsct요소에 있는 div태그4번쨰-div-div2번
time.sleep(1)
for i in range(15):
    time.sleep(0.5)
    driver.find_element(By.XPATH, button_xpath).click()

titles = []
for i in range(1,98):
    for j in range(1,7):

        title_xpath = '//*[@id="newsct"]/div[4]/div/div[1]/div[{}]/ul/li[{}]/div/div/div[2]/a/strong'.format(i,j)
        try :
            title = driver.find_element(By.XPATH,title_xpath).text
            title = re.compile('[^가-힣 ]').sub('', title)
            titles.append(title)
            #print(title)
        except: #예외처리
            print(i,j)
    df_section_titles = pd.DataFrame(titles, columns=['titles'])  # 데이터프레임 생성
    df_section_titles['category'] = category[0]  # 카테고리 라벨 붙이기
    df_titles = pd.concat([df_titles, df_section_titles], axis='rows', ignore_index=True)  # 빈 데이터프레임에 row정장


print(df_titles.head())
df_titles.info()
print(df_titles['category'].value_counts())
df_titles.to_csv('./crawling_data/naver_headline_news_politics{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False) # 나노second단위 받은 시간으로 오늘 날짜로 바꿔서 저장
time.sleep(30)
driver.close()

