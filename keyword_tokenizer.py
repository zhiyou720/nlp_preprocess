import jieba
import jieba.posseg as jpg
from nltk.corpus import stopwords


class JiebaTokenizer:

    def __init__(self, user_dict_path, puncs_path, considered_tags_path, stop_word_path, ignore_tag='-'):
        self.user_dict_path = user_dict_path
        self.puncs_path = puncs_path
        self.considered_tags_path = considered_tags_path
        self.stop_word_path = stop_word_path
        self.ignore_tag = ignore_tag

        self.load_user_dict()

    def tokenizer(self, text):
        """
        :param text: str
        :return:
        """

        tokens_tagged = [(list(x)[0], list(x)[1]) for x in jpg.lcut(text)]
        # print(tokens_tagged)
        tokens_tagged = self.tag_puncs(tokens_tagged)

        tokens_tagged = self.tag_unconsidered(tokens_tagged)

        tokens_tagged = self.tag_stop_words(tokens_tagged)

        return tokens_tagged

    def load_user_dict(self):
        user_dict = [x.split(' ')[0] for x in self.load_txt_data(self.user_dict_path)]
        for x in user_dict:
            jieba.add_word(x)

    @staticmethod
    def load_txt_data(path, mode='utf-8-sig', origin=False):
        """
        This func is used to reading txt file
        :param origin:
        :param path: path where file stored
        :param mode:
        :type path: str
        :return: string lines in file in a list
        :rtype: list
        """
        if type(path) != str:
            raise TypeError
        res = []

        file = open(path, 'rb')
        lines = file.read().decode(mode, errors='ignore')
        for line in lines.split('\n'):
            line = line.strip()
            if origin:
                res.append(line)
            else:
                if line:
                    res.append(line)
        file.close()
        return res

    def tag_puncs(self, tokens_tagged):
        puncs = set(x for x in self.load_txt_data(self.puncs_path))
        new_tokens_tagged = []
        for i in range(len(tokens_tagged)):
            if tokens_tagged[i][0] not in puncs:
                new_tokens_tagged.append(tokens_tagged[i])
            else:
                new_tokens_tagged.append((tokens_tagged[i][0], self.ignore_tag))

        return new_tokens_tagged

    def tag_stop_words(self, tokens_tagged):
        stop_word = set(self.load_txt_data(self.stop_word_path) + stopwords.words('english'))
        new_tokens_tagged = []

        for i in range(len(tokens_tagged)):
            if tokens_tagged[i][0].lower() in stop_word or tokens_tagged[i][0].lower() == ' ':
                new_tokens_tagged.append((tokens_tagged[i][0], self.ignore_tag))
            else:
                new_tokens_tagged.append(tokens_tagged[i])

        return new_tokens_tagged

    def tag_unconsidered(self, tokens_tagged):
        considered_tags = set(self.load_txt_data(self.considered_tags_path))

        new_tokens_tagged = []
        for i in range(len(tokens_tagged)):
            if tokens_tagged[i][1] in considered_tags:
                new_tokens_tagged.append(tokens_tagged[i])
            else:
                new_tokens_tagged.append((tokens_tagged[i][0], self.ignore_tag))
        return new_tokens_tagged


def filtrate(tagged_tokens, ignore_tag='-'):
    return [x[0] for x in tagged_tokens if x[1] != ignore_tag and x[0] != ' ']


def get_sif_candidates(tagged_tokens, ignore_tag='-'):
    candidates = []
    for i in range(len(tagged_tokens)):
        if tagged_tokens[i][1] != ignore_tag:
            candidates.append((tagged_tokens[i][0], (i, i + 1)))
    return candidates


if __name__ == '__main__':
    _jieba_tokenizer = JiebaTokenizer(user_dict_path='conf/user_dict.txt',
                                      puncs_path='conf/punctuation.dat',
                                      considered_tags_path='conf/jieba_considered_tags.txt',
                                      stop_word_path='conf/stopwords.txt', )

    _tokenize = _jieba_tokenizer.tokenizer

    _text = '今天又给大家送福利啦,现在加入腾讯汽车会员，享中石油、中石化实名充值九折卡，最高100万的驾乘意外险、违章代缴6折优惠等更多！'
    _tagged_tokens = _tokenize(_text)
    __ = filtrate(_tagged_tokens)
    ___ = get_sif_candidates(_tagged_tokens)
    print(__)
    print(___)

    _text = '从君士坦丁堡陷落到地中海海洋混战，一直到威尼斯《海洋霸权》的史诗，其实这三本书的角度和视角并不一致，' \
            '但是总有一种主题和精神气质贯穿始终。基督文明和伊斯兰文明在地中海的碰撞，献上了交织着血与泪的史诗。'
    _tagged_tokens = _tokenize(_text)
    __ = filtrate(_tagged_tokens)
    ___ = get_sif_candidates(_tagged_tokens)
    print(__)
    print(___)

    _text = '(38）对世界的认知是知识，对自己的认知是智慧。知识和智慧是跟随我们一辈子不断成长、永无止境的。'
    _tagged_tokens = _tokenize(_text)
    __ = filtrate(_tagged_tokens)
    ___ = get_sif_candidates(_tagged_tokens)
    print(__)
    print(___)

    _text = '看了个遍，在开始狂铺知识点之前，我们先把时间向前移一点。'
    _tagged_tokens = _tokenize(_text)
    __ = filtrate(_tagged_tokens)
    ___ = get_sif_candidates(_tagged_tokens)
    print(__)
    print(___)

    _text = '《The Song of the Golden Dragon》是Estas Tonne在德国巴伐利亚的兰茨胡特街头的弹奏作品,在网络上评价极高,甚至被网友奉为街头神兽,其精湛的演奏技术和感染力让人叹为观止!'
    _tagged_tokens = _tokenize(_text)
    __ = filtrate(_tagged_tokens)
    ___ = get_sif_candidates(_tagged_tokens)
    print(_tagged_tokens)
    print(__)
    print(___)


