from elmoformanylangs import Embedder
import numpy as np
import torch
import nltk


class InputTextObj:
    """Represent the input text in which we want to extract key-phrases"""

    def __init__(self, tokens, tokens_tagged, candidate):
        """
        :param tokens: ['word', 'word', 'word']
        :param tokens_tagged: [('word', 'tag'), ()]
        """
        self.tokens = tokens
        self.tokens_tagged = tokens_tagged
        if candidate:
            self.key_phrase_candidate = candidate
        else:
            self.key_phrase_candidate = self.extract_candidates(self.tokens_tagged)

    @staticmethod
    def extract_candidates(tokens_tagged):
        """
        Based on part of speech return a list of candidate phrases
        :param tokens_tagged:
        :return key_phrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
        """
        grammar_zh = """  NP:
                {<n.*|a|uw|i|j|x>*<n.*|uw|x>|<x|j><-><m|q>} # Adjective(s)(optional) + Noun(s)"""
        np_parser = nltk.RegexpParser(grammar_zh)  # Noun phrase parser
        key_phrase_candidate = []
        np_pos_tag_tokens = np_parser.parse(tokens_tagged)
        count = 0
        for token in np_pos_tag_tokens:
            if isinstance(token, nltk.tree.Tree) and token._label == "NP":
                _np = ''.join(word for word, tag in token.leaves())
                length = len(token.leaves())
                start_end = (count, count + length)
                count += length
                key_phrase_candidate.append((_np, start_end))

            else:
                count += 1

        return key_phrase_candidate


class WordEmbeddings:
    """
        ELMo
        https://allennlp.org/elmo

    """

    def __init__(self, model_path):
        self.elmo = Embedder(model_path)

    def get_tokenized_words_embeddings(self, sents_tokened):
        """
        @see EmbeddingDistributor
        :param sents_tokened:
        :return: ndarray with shape (len(sents), dimension of embeddings)
        """
        max_len = max([len(sent) for sent in sents_tokened])
        elmo_embedding = self.elmo.sents2elmo(sents_tokened, output_layer=-2)
        elmo_embedding = [np.pad(emb, pad_width=((0, 0), (0, max_len - emb.shape[1]), (0, 0)), mode='constant') for emb
                          in elmo_embedding]
        elmo_embedding = torch.from_numpy(np.array(elmo_embedding))
        return elmo_embedding


class SentEmbeddings:

    def __init__(self,
                 word_embeddor,
                 weight_file_pretrain=r'SIFRank_zh/auxiliary_data/dict.txt',
                 weight_file_finetune=r'SIFRank_zh/auxiliary_data/dict.txt',
                 weight_para_pretrain=2.7e-4,
                 weight_para_finetune=2.7e-4,
                 ignore_tag='-'):

        self.word2weight_pretrain = self.get_word_weight(weight_file_pretrain, weight_para_pretrain)
        self.word2weight_finetune = self.get_word_weight(weight_file_finetune, weight_para_finetune)

        self.word_embeddor = word_embeddor
        self.english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        self.chinese_punctuations = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、“〃《》「」『』【】〔〕〖〗' \
                                    '〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'
        self.ignore_tag = ignore_tag

    def get_tokenized_sent_embeddings(self, text_obj):
        """
        Based on part of speech return a list of candidate phrases
        """

        sent = text_obj.tokens
        sent_tagged = text_obj.tokens_tagged
        key_phrase_candidate = text_obj.key_phrase_candidate

        tokens_segmented = self.get_sent_segmented(sent)
        elmo_embeddings = self.word_embeddor.get_tokenized_words_embeddings(tokens_segmented)
        elmo_embeddings = self.context_embeddings_alignment(elmo_embeddings, tokens_segmented)
        elmo_embeddings = self.splice_embeddings(elmo_embeddings, tokens_segmented)

        weight_list = self.get_weight_list(self.word2weight_pretrain, sent, sent_tagged)

        sent_embeddings = self.get_weighted_average(sent, sent_tagged, weight_list, elmo_embeddings[0])

        candidate_embeddings_list = []
        for kc in key_phrase_candidate:
            start = kc[1][0]
            end = kc[1][1]
            kc_emb = self.get_candidate_weighted_average(sent, weight_list, elmo_embeddings[0], start, end)
            candidate_embeddings_list.append(kc_emb)

        return sent_embeddings, candidate_embeddings_list

    @staticmethod
    def get_sent_segmented(tokens):
        min_seq_len = 16
        sents_sectioned = []
        if len(tokens) <= min_seq_len:
            sents_sectioned.append(tokens)
        else:
            position = 0
            for i, token in enumerate(tokens):
                if token == '.' or token == '。':
                    if i - position >= min_seq_len:
                        sents_sectioned.append(tokens[position:i + 1])
                        position = i + 1
            if len(tokens[position:]) > 0:
                sents_sectioned.append(tokens[position:])

        return sents_sectioned

    @staticmethod
    def context_embeddings_alignment(elmo_embeddings, tokens_segmented):
        """
        Embeddings Alignment
        :param elmo_embeddings: The embeddings from elmo
        :param tokens_segmented: The list of tokens list
         <class 'list'>: [['今', '天', '天气', '真', '好', '啊'],['潮水', '退', '了', '就', '知道', '谁', '没', '穿', '裤子']]
        :return:
        """
        token_emb_map = {}
        n = 0
        for i in range(0, len(tokens_segmented)):

            for j, token in enumerate(tokens_segmented[i]):

                emb = elmo_embeddings[i, 1, j, :]
                if token not in token_emb_map:
                    token_emb_map[token] = [emb]
                else:
                    token_emb_map[token].append(emb)
                n += 1

        anchor_emb_map = {}
        for token, emb_list in token_emb_map.items():
            average_emb = emb_list[0]
            for j in range(1, len(emb_list)):
                average_emb += emb_list[j]
            average_emb /= float(len(emb_list))
            anchor_emb_map[token] = average_emb

        for i in range(0, elmo_embeddings.shape[0]):
            for j, token in enumerate(tokens_segmented[i]):
                emb = anchor_emb_map[token]
                elmo_embeddings[i, 2, j, :] = emb

        return elmo_embeddings

    @staticmethod
    def splice_embeddings(elmo_embeddings, tokens_segmented):
        new_elmo_embeddings = elmo_embeddings[0:1, :, 0:len(tokens_segmented[0]), :]
        for i in range(1, len(tokens_segmented)):
            emb = elmo_embeddings[i:i + 1, :, 0:len(tokens_segmented[i]), :]
            new_elmo_embeddings = torch.cat((new_elmo_embeddings, emb), 2)
        return new_elmo_embeddings

    def get_weight_list(self, word2weight_pretrain, tokenized_sents, sent_tagged):
        weight_list = []
        for i in range(len(tokenized_sents)):
            word = tokenized_sents[i]
            word = word.lower()
            tag = sent_tagged[i][1]
            weight_pretrain = self.get_oov_weight(tokenized_sents, word2weight_pretrain, word, tag)
            weight = weight_pretrain

            weight_list.append(weight)

        return weight_list

    def get_weighted_average(self, sent, sents_tokened_tagged, weight_list, embeddings_list):
        num_words = len(sent)
        assert num_words == len(weight_list)
        sum_w = torch.zeros((3, 1024))
        for i in range(0, 3):
            for j in range(0, num_words):
                if sents_tokened_tagged[j][1] != self.ignore_tag:
                    e_test = embeddings_list[i][j]
                    sum_w[i] += e_test * weight_list[j]
            sum_w[i] = sum_w[i] / float(num_words)
        return sum_w

    @staticmethod
    def get_word_weight(weight_file="", weight_para=2.7e-4):
        """
        Get the weight of words by word_fre/sum_fre_words
        :param weight_file
        :param weight_para
        :return: word2weight[word]=weight : a dict of word weight
        """
        if weight_para <= 0:  # when the parameter makes no sense, use unweighted
            weight_para = 1.0
        word2weight = {}
        word2fre = {}
        with open(weight_file, encoding='UTF-8') as f:
            lines = f.readlines()
        # sum_num_words = 0
        sum_fre_words = 0
        for line in lines:
            word_fre = line.split()
            # sum_num_words += 1
            if len(word_fre) >= 2:
                word2fre[word_fre[0]] = float(word_fre[1])
                sum_fre_words += float(word_fre[1])
            else:
                print(line)
        for key, value in word2fre.items():
            word2weight[key] = weight_para / (weight_para + value / sum_fre_words)
            # word2weight[key] = 1.0 #method of RVA
        return word2weight

    @staticmethod
    def get_candidate_weighted_average(sent, weight_list, embeddings_list, start, end):
        assert len(sent) == len(weight_list)
        num_words = end - start
        sum_w = torch.zeros((3, 1024))
        for i in range(0, 3):
            for j in range(start, end):
                e_test = embeddings_list[i][j]
                sum_w[i] += e_test * weight_list[j]
            sum_w[i] = sum_w[i] / float(num_words)
        return sum_w

    def get_oov_weight(self, tokenized_sents, word2weight, word, tag):
        wnl = nltk.WordNetLemmatizer()
        word = wnl.lemmatize(word)

        if word in word2weight:  #
            return word2weight[word]

        if word in self.english_punctuations or word in self.chinese_punctuations:  # The oov_word is a punctuation
            return 0.0
        if tag == self.ignore_tag:
            return 0.0

        max_w = 0.0
        for w in tokenized_sents:
            if w in word2weight and word2weight[w] > max_w:
                max_w = word2weight[w]
        return max_w  # Return the max weight of word in the tokenized_sents


class SIFRank:
    def __init__(self, elmo_layers_weight=None):

        if elmo_layers_weight:
            self.elmo_layers_weight = elmo_layers_weight
        else:
            self.elmo_layers_weight = [0.0, 1.0, 0.0]

    def compute_rank(self, text_obj, sent_embeddings_obj, top_k=15):
        """
        :param text_obj:
        :param sent_embeddings_obj:
        :param top_k:
        :return:
        """
        sent_embeddings, candidate_embeddings_list = sent_embeddings_obj.get_tokenized_sent_embeddings(text_obj)

        dist_list = []
        for i, word_embedding in enumerate(candidate_embeddings_list):
            # TODO: cosine sim
            dist = self.get_dist_cosine(sent_embeddings, word_embedding)
            dist_list.append(dist)

        dist_all = self.get_all_dist(candidate_embeddings_list, text_obj.key_phrase_candidate, dist_list)
        dist_final = self.get_final_dist(dist_all)
        dist_sorted = sorted(dist_final.items(), key=lambda x: x[1], reverse=True)
        return dist_sorted[0:top_k]

    def compute_rank2(self, text_obj, sent_embeddings_obj, top_k=15, position_bias=3.4):
        """
        :param position_bias:
        :param sent_embeddings_obj:
        :param text_obj:
        :param top_k: the top-N number of keyphrases
        :return:
        """
        sent_embeddings, candidate_embeddings_list = sent_embeddings_obj.get_tokenized_sent_embeddings(text_obj)
        position_score = self.get_position_score(text_obj.key_phrase_candidate, position_bias)
        average_score = sum(position_score.values()) / float(len(position_score))

        dist_list = []
        for i, word_embedding in enumerate(candidate_embeddings_list):
            dist = self.get_dist_cosine(sent_embeddings, word_embedding)
            dist_list.append(dist)
        dist_all = self.get_all_dist(candidate_embeddings_list, text_obj.key_phrase_candidate, dist_list)
        dist_final = self.get_final_dist(dist_all)

        for _np, dist in dist_final.items():
            if _np in position_score:
                dist_final[_np] = dist * position_score[_np] / average_score
        dist_sorted = sorted(dist_final.items(), key=lambda x: x[1], reverse=True)
        return dist_sorted[0:top_k]

    def get_position_score(self, keyphrase_candidate_list, position_bias):
        position_score = {}
        for i, kc in enumerate(keyphrase_candidate_list):
            _np = kc[0]
            _np = _np.lower()
            wnl = nltk.WordNetLemmatizer()
            _np = wnl.lemmatize(_np)
            if _np in position_score:

                position_score[_np] += 0.0
            else:
                position_score[_np] = 1 / (float(i) + 1 + position_bias)
        score_list = []
        for _np, score in position_score.items():
            score_list.append(score)
        score_list = self.softmax(score_list)

        i = 0
        for _np, score in position_score.items():
            position_score[_np] = score_list[i]
            i += 1
        return position_score

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def get_dist_cosine(self, emb1, emb2):
        sum_w = 0.0
        assert emb1.shape == emb2.shape
        n = len(self.elmo_layers_weight)
        for i in range(0, n):
            a = emb1[i]
            b = emb2[i]
            sum_w += self.cos_sim(a, b) * self.elmo_layers_weight[i]
        return sum_w

    @staticmethod
    def cos_sim(vector_a, vector_b):
        """
        计算两个向量之间的余弦相似度
        :param vector_a: 向量 a
        :param vector_b: 向量 b
        :return: sim
        """
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denom == 0.0:
            return 0.0
        else:
            cos = num / denom
            sim = 0.5 + 0.5 * cos
            return sim

    @staticmethod
    def get_all_dist(candidate_embeddings_list, key_phrase_candidate, dist_list):
        """
        :param candidate_embeddings_list:
        :param key_phrase_candidate:
        :param dist_list:
        :return: dist_all
        """

        dist_all = {}
        for i, emb in enumerate(candidate_embeddings_list):
            phrase = key_phrase_candidate[i][0]
            phrase = phrase.lower()
            wml = nltk.WordNetLemmatizer()
            phrase = wml.lemmatize(phrase)
            if phrase in dist_all:
                # store the No. and distance
                dist_all[phrase].append(dist_list[i])
            else:
                dist_all[phrase] = []
                dist_all[phrase].append(dist_list[i])
        return dist_all

    @staticmethod
    def get_final_dist(dist_all):
        """
        :param dist_all:
        :return:
        """

        final_dist = {}

        for phrase, dist_list in dist_all.items():
            sum_dist = 0.0
            for dist in dist_list:
                sum_dist += dist
            final_dist[phrase] = sum_dist / float(len(dist_list))
        return final_dist


def keyword_sif_rank(tokens, tokens_tagged, candidates, sent_embeddings_obj, num_words=100):
    text_obj = InputTextObj(tokens, tokens_tagged, candidate=candidates)

    keyword_ranker = SIFRank()
    keyword = keyword_ranker.compute_rank2(text_obj=text_obj,
                                           sent_embeddings_obj=sent_embeddings_obj,
                                           top_k=num_words)

    keyword = [x for x in keyword if x]

    return keyword


if __name__ == '__main__':
    # _text = '今天又给大家送福利啦,现在加入腾讯汽车会员，享中石油、中石化实名充值九折卡，最高100万的驾乘意外险、违章代缴6折优惠等更多！'
    _text = '一同来到节目的还有这首歌的创作者，同样是 TZ 签约艺人的实力唱将李林。'
    from seg_tokenizer import _tokenize, get_sif_candidates

    model_file = r'conf/sif_model/zhs.model/'
    _elmo_embeddings = WordEmbeddings(model_file)

    _tokens_tagged = _tokenize(_text)
    _tokens = [x[0] for x in _tokens_tagged]
    _candidates = get_sif_candidates(_tokens_tagged)
    print(_tokens)

    _ = keyword_sif_rank(_tokens, _tokens_tagged, candidates=_candidates)
    print(_)
