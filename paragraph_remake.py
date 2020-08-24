def count_info(data):
    """
    返回有用的文字数量
    :param data:
    :return:
    """
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', ' ']
    chinese_punctuations = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'
    puncs = set(english_punctuations + list(chinese_punctuations))
    i = 0
    for char in data:
        if char not in puncs:
            i += 1
    return i


def check_stack(stack):
    count = 0
    for i in range(len(stack)):
        count += len(stack[i])
    if count <= 128:
        return True
    else:
        return False


def data_re_combine(data):
    """

    :param data: [sent1, sent2]
    :return:
    """
    stack = []
    new_data = []
    i = 0
    while i < len(data):
        useful_info = count_info(data[i])
        if 64 < useful_info <= 128:
            new_data.append(data[i])
            i += 1
        elif useful_info < 64:

            stack.append(data[i])
            if check_stack(stack):
                i += 1
            else:
                new_data.append(''.join(stack))
                stack = []
                i += 1
        else:
            new_data.append(data[i])
            i += 1
    if stack:
        new_data.append(''.join(stack))
    return new_data


if __name__ == '__main__':
    _path = 'data/db_data.1.json'
    from dataio import load_json_data
    from h5_preprocessor import process_html_tag

    _data_set = load_json_data(_path, json_data_name='RECORDS', data_num=10)
    for item in _data_set:
        item = process_html_tag(item['content'])
        print(item)
        print(data_re_combine(item))
