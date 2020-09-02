import gensim
import numpy as np


class WordEmbedding(object):
    """
    """

    def __init__(self, model_name):
        self.w2v_model = gensim.models.KeyedVectors.load(model_name, mmap='r')

    @staticmethod
    def _compute_ngrams(word, min_n, max_n):
        extended_word = word
        ngrams = []
        for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
            for i in range(0, len(extended_word) - ngram_length + 1):
                ngrams.append(extended_word[i:i + ngram_length])
        return list(set(ngrams))

    def most_similar(self, word, topn=10):
        return self.w2v_model.most_similar(positive=word, topn=topn)

    def word_vector(self, word, min_n=1, max_n=3):
        word_size = self.w2v_model.vectors[0].shape[0]  # 确认词向量维度
        ngrams = self._compute_ngrams(word, min_n=min_n, max_n=max_n)  # 计算word的ngrams词组
        if word in self.w2v_model.index2word:  # exist
            return self.w2v_model[word]
        else:
            word_vec = np.zeros(word_size, dtype=np.float32)  # 计算与词相近的词向量
            ngrams_found = 0
            ngrams_single = [ng for ng in ngrams if len(ng) == 1]  # 单个字的词
            ngrams_more = [ng for ng in ngrams if len(ng) > 1]  # 多个字的词
            for ngram in ngrams_more:  # 优先考虑多字词
                if ngram in self.w2v_model.index2word:
                    word_vec += self.w2v_model[ngram]
                    ngrams_found += 1
            if ngrams_found == 0:  # 否则考虑单字词
                for ngram in ngrams_single:
                    if ngram in self.w2v_model.index2word:
                        word_vec += self.w2v_model[ngram]
                        ngrams_found += 1
            if word_vec.any():  # 只要有一个不为0
                return word_vec / max(1, ngrams_found)
            else:
                # TODO: return 0
                return word_vec / max(1, ngrams_found)

    def embedding(self, keywords):
        embedding = np.zeros((len(keywords), 200))
        if keywords is None or len(keywords) == 0:
            pass
        else:
            for i, kw in enumerate(keywords):
                embedding[i, :] = self.word_vector(kw)
        return embedding


if __name__ == "__main__":
    m = WordEmbedding("models/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.bin")
    m.w2v_model.most_similar("中")
    s = m.w2v_model.similarity("杨振宁", "毛泽东")
    print(s)
    _words = ['福利', '腾讯', '汽车', '会员', '中石油', '中石化', '充值', '九折卡', '意外险', '违章']
    for _word in _words:
        print(m.word_vector(_word))
