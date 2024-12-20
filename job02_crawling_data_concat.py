import pandas as pd
import os

# crawling_data 폴더 경로
folder_path = 'crawling_data'

# 폴더 내 모든 CSV 파일 경로 가져오기
csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# 모든 CSV 파일을 읽고 리스트에 저장
dataframes = [pd.read_csv(file) for file in csv_files]

# 모든 데이터프레임을 하나로 합치기
combined_df = pd.concat(dataframes, ignore_index=True)

# 결과를 새로운 CSV 파일로 저장
combined_df.to_csv('naver_headline_news_20241219_combined_data.csv', index=False)