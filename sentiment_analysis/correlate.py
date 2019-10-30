import numpy as np # linear algebra
import pandas as pd # test processing, CSV file I/O (e.g. pd.read_csv)

import csv


def main():
    tweet_file = "results/score.csv"
    stock_file = "../data/stocks_data/change.csv"
    tweet_predic = pd.read_csv(tweet_file, encoding='latin-1')
    stock = pd.read_csv(stock_file, encoding='latin-1')

    print(tweet_predic)
    print(stock)
    merged_data = pd.merge(tweet_predic, stock, left_on="File", right_on="File")
    print(merged_data)
    results = {}
    for time in ["10", "11", "12", "13", "14", "15"]:
        temp = np.corrcoef(merged_data["ratio neg/pos"], merged_data[time])
        results[time] = temp

    for time in results:
        print(time, results[time])

if __name__ == "__main__":
    main()
