import pandas as pd
import os
from datetime import timedelta
from sqlalchemy import create_engine
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# This is the Stocktwits database built by us, the demo of some record in this dataset can be seen in stocktwits_demo.zip
engine = create_engine('mysql+pymysql://root: password@localhost/stocktwits_database', echo=False)
pd.set_option('display.max_columns', None)


folder_path = 'prediction_accuracy_reports'
os.makedirs(folder_path, exist_ok=True)
# 加载数据
query = "SELECT * FROM LatestUserTweets2 ORDER BY user_id, tweet_time"
latest_tweets = pd.read_sql(query, engine)
print(latest_tweets)
latest_tweets['tweet_time'] = pd.to_datetime(latest_tweets['tweet_time'])
latest_tweets['is_correct'] = (latest_tweets['tweet_sentiment'] == latest_tweets['gt_sentiment']).astype(int)


def calculate_long_short_term_accuracy(df):
    df = df.sort_values('tweet_time')
    df['tweet_day'] = df['tweet_time'].dt.date
    results = pd.DataFrame()

    for idx, row in df.iterrows():
        two_years_ago = row['tweet_time'] - timedelta(days=730)
        past_tweets = df[(df['tweet_time'] >= two_years_ago) & (df['tweet_time'] < row['tweet_time'])]

        day=row['tweet_day']
        past_data = df[df['tweet_day'] < day]

        last_20 = past_data.tail(20)
        # if (last_20['tweet_day'].nunique() >= 5) and (len(last_20) == 20) and (last_20['stock'].nunique() <= 5):
        if (last_20['tweet_day'].nunique() >= 5) and (len(last_20)==20):
            accuracy_last_20 = last_20['is_correct'].mean()
        else:
            accuracy_last_20=0.5
        accuracy_last_two_year = past_tweets['is_correct'].mean()

        if accuracy_last_20 >= 0.8 or accuracy_last_20 <= 0.2:
            row['accuracy']=accuracy_last_20
            row['acc_last_2_years'] = accuracy_last_two_year
            number = past_tweets['tweet_time'].nunique()
            row['past_num'] = number
            earliest_day = past_tweets['tweet_time'].min()
            earliest_day_20=last_20['tweet_time'].min()
            day_diff_20=(row['tweet_time'] - earliest_day_20).days
            day_diff = (row['tweet_time'] - earliest_day).days
            row['diff_time'] = day_diff
            row['diff_time_20']=day_diff_20
            row['action'] = 'follow' if accuracy_last_20 >= 0.75 else 'opposite'
            row['predicted_correct'] = row['tweet_sentiment'] == row['gt_sentiment'] if row[ 'action'] == 'follow' else row['tweet_sentiment'] != row['gt_sentiment']
            row = row.to_frame().T
            results = pd.concat([results, row])

    return results

batch_counter = 0
results = pd.DataFrame()

for group, group_df in tqdm(latest_tweets.groupby('user_id')):
    results = pd.concat([results, calculate_long_short_term_accuracy(group_df)])
    batch_counter += 1

    if batch_counter % 10000 == 0:
        results_file = os.path.join(folder_path, f'user_predictions_with_{batch_counter // 10000}.csv')
        results.to_csv(results_file, index=False)

results_file = os.path.join(folder_path, f'user_predictions_with_trade_info_new.csv')
results.to_csv(results_file, index=False)

print("Predictions and their accuracies have been saved.")

def adjust_sentiment(row):
    if row['action'] == 'opposite':
        if row['tweet_sentiment'] == 'Bullish':
            return 'Bearish'
        elif row['tweet_sentiment'] == 'Bearish':
            return 'Bullish'
    return row['tweet_sentiment']


file = results_file

df = pd.read_csv(file)
df['tweet_time'] = pd.to_datetime(df['tweet_time'])

# 筛选时间在 2019 年到 2023 年之间的记录
df_filtered = df[(df['tweet_time'] >= '2019-01-01') & (df['tweet_time'] <= '2023-12-31')]

df_filtered['predicted_correct'] = df_filtered['predicted_correct'].astype(int)

df_filtered_result = df_filtered[
    ((df_filtered['accuracy'] <= 0.2) & (df_filtered['acc_last_2_years'] <= 0.35)) |
    ((df_filtered['accuracy'] >= 0.8) & (df_filtered['acc_last_2_years'] >= 0.65))
]


df_combine=df_filtered_result.groupby(['stock', 'stock_time']).apply(lambda x: x.sample(n=1)).reset_index(drop=True)


df_combine['pseudo_gt'] = df_combine.apply(adjust_sentiment, axis=1)
print(df_combine)
df_combine = df_combine.filter(['stock', 'stock_time', 'gt_sentiment', 'pseudo_gt'])


csv_file_path = 'psudo_combine_all.csv'  
df_combine.to_csv(csv_file_path, index=False)

