import re
from bs4 import BeautifulSoup
import unicodedata


def process_html_tag(data):
    """

    :param data: str
    :return: str
    """
    if not data:
        return None
    soup = BeautifulSoup(data, 'html.parser')
    data = soup.get_text()
    data = unicodedata.normalize('NFKC', data)

    data = re.sub('\n+', '\n', data)
    data = re.sub(' +', ' ', data)
    data = re.sub('\t', ' ', data)
    data = [x.strip() for x in data.split('\n') if x.strip()]
    if data:
        return data
    else:
        return None


if __name__ == '__main__':
    _path = 'data/db_data.1.json'
    from dataio import load_json_data

    _data_set = load_json_data(_path, json_data_name='RECORDS', data_num=10)
    for _data in _data_set:
        content = _data['content']
        content = process_html_tag(content)
        print(content)
        print('*' * 128)
    # _res, _max_len = process_html_tag()
    # print(_max_len)
    # print(len(_res))
