from keyword_tokenizer import filtrate, get_sif_candidates
from cluster import kmeans_key_word
from SIF import keyword_sif_rank


class KeyWordExt:
    def __init__(self,
                 tokenizer,
                 tencent_embeddor,
                 elmo_sent_embeddor,
                 random_seed,
                 sif=False
                 ):
        self.tokenizer = tokenizer  # 实例外预加载
        self.tencent_embeddor = tencent_embeddor  # 实例外预加载
        self.elmo_sent_embeddor = elmo_sent_embeddor  # 实例外预加载

        self.sif = sif

        self.random_seed = random_seed

    def extract(self, text, max_k, top_n):

        tagged_tokens = self.tokenizer(text)  # 运算 开销不计
        filter_tokens = filtrate(tagged_tokens)  # 运算 开销不计
        unique_filter_tokens = list(set(filter_tokens))  # 运算 开销不计

        if not unique_filter_tokens:
            # no key word
            return None
        elif len(unique_filter_tokens) == 1:
            # one key word
            return unique_filter_tokens

        elif len(unique_filter_tokens) == 2:
            # two key word
            return unique_filter_tokens
        elif len(unique_filter_tokens) == 3:
            if self.sif:
                sif_candidates = get_sif_candidates(tagged_tokens)  # 运算 开销不计
                tokens = [x[0] for x in tagged_tokens]  # 运算 开销不计

                key_words = keyword_sif_rank(tokens,
                                             tagged_tokens,
                                             candidates=sif_candidates,
                                             sent_embeddings_obj=self.elmo_sent_embeddor
                                             )[:2]  # 运算 开销大

                return [x[0] for x in key_words]
            else:
                return self.extract_k_means(filter_tokens, max_k=2, top_n=top_n)

        else:
            return self.extract_k_means(filter_tokens, max_k=max_k, top_n=top_n)

    def tencent_embedding(self, tokens):
        vectors = []

        for token in tokens:
            vectors.append(self.tencent_embeddor(token))
        return vectors

    def extract_k_means(self, filter_tokens, max_k, top_n):
        try:
            import time
            start = time.time()
            vectors = self.tencent_embedding(filter_tokens)
            end = time.time()
            # print("Embedding used {}s".format(end - start))

            key_word_set = kmeans_key_word(words=filter_tokens, vectors=vectors,
                                           max_k=max_k, top_n=top_n, random_seed=self.random_seed)  # 运算 开销大
            key_words = [y for x in key_word_set for y in x]
            return key_words

        except Exception as e:
            raise e


if __name__ == '__main__':
    from keyword_tokenizer import JiebaTokenizer
    from tencent_word_embedding import WordEmbedding
    from SIF import SentEmbeddings

    _jieba_tokenizer = JiebaTokenizer(user_dict_path='conf/user_dict.txt',
                                      puncs_path='conf/punctuation.dat',
                                      considered_tags_path='conf/jieba_considered_tags.txt',
                                      stop_word_path='conf/stopwords.txt', )

    _tokenizer = _jieba_tokenizer.tokenizer

    _sent_embeddings_obj = SentEmbeddings(weight_file_pretrain='conf/sif_model/dict.txt',
                                          weight_file_finetune='conf/sif_model/dict.txt')

    _tencent_word_embedding = WordEmbedding("conf/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.bin")
    _tencent_embeddor = _tencent_word_embedding.word_vector

    print("LOAD_SUCCESSFUL")
    _text = '有啥好得瑟的车主就忍不住了,你见过一百万的破车吗？最后还是加油站的站长出面同意把汽油吸出来，并且赔偿结果没成。'
    _extractor = KeyWordExt(tokenizer=_tokenizer,
                            tencent_embeddor=_tencent_embeddor,
                            elmo_sent_embeddor=_sent_embeddings_obj)
    _key_word = _extractor.extract(_text, max_k=3, top_n=2)
    print(_key_word)
