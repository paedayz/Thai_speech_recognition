import __main__

word_df = __main__.pd.read_csv('important_and_stop_words.csv', index_col=False)
important_words = word_df[word_df['Is not important'] == 1]['Word'].values
stop_words = word_df[word_df['Is not important'] == 0]['Word'].values