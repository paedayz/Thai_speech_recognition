import __main__
from tokenizer import tokenize

def generate_word_frequency_report(text):
    text_list = tokenize(text)

    filter_important_text = [i for i in text_list if i in __main__.important_words]
    filter_out_stop_word_text = [i for i in text_list if i not in __main__.stop_words and i not in filter_important_text and len(i) > 1]

    text_dict = {}
    for word in filter_important_text + filter_out_stop_word_text:
        if word not in text_dict:
            text_dict[word] = text.count(word)

    sort_dict = dict(reversed(sorted(text_dict.items(), key=lambda item: item[1])))

    result_df = __main__.pd.DataFrame(sort_dict.items(), columns=['word', 'frequency'])
    result_df.to_csv('report.csv', encoding='utf-8')

    print('save report successfully')