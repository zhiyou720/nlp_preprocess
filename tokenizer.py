import re
from dataio import load_txt_data

# TODO: 需要和被预测的标点保持一致, 这里列出的比较全面
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
chinese_punctuations = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、“〃《》「」『』【】〔〕〖〗' \
                       '〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'

PUNCS = [x for x in chinese_punctuations] + english_punctuations
PUNCS = ['\\' + x for x in PUNCS]
PUNC_TAGS = [x.split('\t')[1] for x in load_txt_data('model/config/punctuation.dat')]
PUNC_TAGS = ['\\' + x for x in PUNC_TAGS]


def spliter(string):
    english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    output = []
    buffer = ''
    for s in string:
        if s in english or s in english.upper():
            buffer += s
        else:
            if buffer:
                output.append(buffer)
            buffer = ''
            output.append(s)
    if buffer:
        output.append(buffer)

    while ' ' in output:
        output.remove(' ')

    return output


def remove_puncs(data):
    for punc in PUNC_TAGS:
        data = re.sub(punc, '', data)
    return data


def tokenize(data):
    """
    :param data: str
    :return: list
    """
    if not data:
        return None

    data = spliter(data)
    data = ' '.join(data)
    data = remove_puncs(data)
    data = [x for x in data.split(' ') if x]

    if not data:
        return None
    else:
        return data
