from keyword_tokenizer import JiebaTokenizer
from tencent_word_embedding import WordEmbedding
from SIF import SentEmbeddings, WordEmbeddings
from extraction import KeyWordExt


# _text = '有啥好得瑟的车主就忍不住了,你见过一百万的破车吗？最后还是加油站的站长出面同意把汽油吸出来，并且赔偿结果没成。'
# _key_word = key_word_func(_text)
# print(_key_word)


# def key_word_processor(key_word_func, data_batch, true_cpu_thread=4):
#     p = Pool(true_cpu_thread)
#     job = p.map_async(key_word_func, data_batch)
#     return job.get()

# user_dict_path = 'conf/user_dict.txt',
# puncs_path = 'conf/punctuation.dat',
# considered_tags_path = 'conf/jieba_considered_tags.txt',
# stop_word_path = 'conf/stopwords.txt',
# weight_file_pretrain = 'conf/sif_model/dict.txt',
# weight_file_finetune = 'conf/sif_model/dict.txt'
# "conf/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.bin"

class KeyWordExtractor:
    def __init__(self,
                 user_word_path,
                 puncs_path,
                 considered_tags_path,
                 stop_word_path,
                 sif_embedding_path,
                 sif_weight_path,
                 tencent_embedding_path,
                 random_seed
                 ):
        jieba_tokenizer = JiebaTokenizer(user_dict_path=user_word_path,
                                         puncs_path=puncs_path,
                                         considered_tags_path=considered_tags_path,
                                         stop_word_path=stop_word_path, )

        self.tokenizer = jieba_tokenizer.tokenizer

        self.elmo_embeddings = WordEmbeddings(sif_embedding_path)

        self.sent_embeddings_obj = SentEmbeddings(word_embeddor=self.elmo_embeddings,
                                                  weight_file_pretrain=sif_weight_path,
                                                  weight_file_finetune=sif_weight_path)

        tencent_word_embedding = WordEmbedding(tencent_embedding_path)
        self.tencent_embeddor = tencent_word_embedding.word_vector

        # print("LOAD_SUCCESSFUL")
        self.extractor = KeyWordExt(tokenizer=self.tokenizer,
                                    tencent_embeddor=self.tencent_embeddor,
                                    elmo_sent_embeddor=self.sent_embeddings_obj,
                                    random_seed=random_seed)

        self.key_word_func = self.extractor.extract


if __name__ == '__main__':

    from dataio import save_txt_file
    import sys
    import time
    import json
    import random

    _tencent_embedding_path = "conf/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.bin"
    _extractor = KeyWordExtractor(user_word_path='conf/user_dict.txt',
                                  puncs_path='conf/punctuation.dat',
                                  considered_tags_path='conf/jieba_considered_tags.txt',
                                  stop_word_path='conf/stopwords.txt',
                                  sif_embedding_path='conf/sif_model/zhs.model/',
                                  sif_weight_path='conf/sif_model/dict.txt',
                                  tencent_embedding_path=_tencent_embedding_path,
                                  random_seed=1024
                                  )

    key_word_func = _extractor.key_word_func


    def exp_load_data(path='data/res0_150.json'):
        """LOAD DATA"""
        data_set = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['DATA']
            for _i in range(len(data)):

                content = data[_i]['format_content']

                if type(content) == str:
                    content = content.split('\n')
                    for item in content:
                        data_set.append(item)
                elif type(content) == list:
                    if content == ['N']:
                        content = None
                        data[_i]['format_content'] = None
                    else:
                        raise ValueError
                else:
                    raise TypeError
        return data_set


    _data_set = exp_load_data()
    random.seed(1024)
    _data_set = random.sample(_data_set, 50000)

    res = []
    samples_number = len(_data_set)
    start = time.time()

    for i in range(samples_number):
        one_start = time.time()
        key_word = key_word_func(_data_set[i], max_k=3, top_n=2)
        res.append(_data_set[i])
        if key_word:
            res.append(key_word)
        else:
            res.append('NONE')
        # print(_data_set[i])
        # print(key_word)
        res.append('***')
        res.append('***')
        one_end = time.time()
        one_use = one_end - one_start
        total_use = one_end - start
        past_number = i + 1
        mean_use = total_use / past_number
        rest_number = samples_number - past_number
        predict_rest_time = rest_number * mean_use
        sys.stdout.write('\rprocessed: {}/{}, time: {}s/{}s, '
                         'time (h): {}h/{}h, '
                         'mean time {}s'.format(past_number,
                                                samples_number,
                                                round(total_use, 2),
                                                round(predict_rest_time, 2),
                                                round(total_use / 3600, 4),
                                                round(predict_rest_time / 3600, 2),
                                                round(mean_use, 4)
                                                ))
        sys.stdout.flush()

    save_txt_file(res, 'kw.txt')
