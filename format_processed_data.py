import re


def add_space4en(s):
    """
    :param s: ['chr', 'abcd', '123']
    :return:
    """
    res = []
    en = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(len(s)):
        if len(s[i]) != 1:
            if s[i][0].lower() in en:
                res.append(' ' + s[i] + ' ')
                continue
        res.append(s[i])
    return res


def token2sent(data):
    """
    input: [
              [chr, chr, chr]
            ]
    :param data:
    :return: [
                [sent]
                [sent]
            ]
    """
    res = []
    for i in range(len(data)):
        sent = ''.join(add_space4en(data[i]))
        sent = re.sub(' +', ' ', sent).lstrip(' ').rstrip(' ')
        res.append(sent)
    return res
