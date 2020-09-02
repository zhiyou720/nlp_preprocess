import re


# # TODO: 需要和被预测的标点保持一致, 这里列出的比较全面
# english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
# chinese_punctuations = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、“〃《》「」『』【】〔〕〖〗' \
#                        '〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'
# PUNCS = [x for x in chinese_punctuations] + english_punctuations
# PUNCS = ['\\' + x for x in PUNCS]


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


def remove_puncs(data, punc_tags):
    punc_tags = ['\\' + x for x in punc_tags]
    for punc in punc_tags:
        data = re.sub(punc, '', data)
    return data


def tokenize(data, punc_tags):
    """
    :param punc_tags:
    :param data: str # TODO: transform to list data
    :return: list
    """
    if not data:
        return None

    data = spliter(data)
    data = ' '.join(data)
    data = remove_puncs(data, punc_tags)
    data = [x for x in data.split(' ') if x]

    if not data:
        return None
    else:
        return data
