from seg_main import process as seg_process


class PipLine:
    def __init__(self,
                 seg_processor,
                 sentence_transformer,
                 key_word_func,
                 keyword_max_k=3,
                 keyword_top_n=1,
                 summary_max_k=8,
                 summary_top_n=3
                 ):

        self.seg_processor = seg_processor
        self.key_word_func = key_word_func
        self.sentence_transformer = sentence_transformer

        self.keyword_max_k = keyword_max_k
        self.keyword_top_n = keyword_top_n
        self.summary_max_k = summary_max_k
        self.summary_top_n = summary_top_n

    def get_paragraph(self, data):
        return seg_process(data=data, processor=self.seg_processor)

    def get_keyword(self, data):
        if data:
            return self.key_word_func(data, max_k=self.keyword_max_k, top_n=self.keyword_top_n)
        else:
            return None

    def get_keyword_summary(self, data):
        if data:
            return self.key_word_func(data, max_k=self.summary_max_k, top_n=self.summary_top_n)
        else:
            return None

    def get_summary_embedding(self, data):
        if data:
            return self.sentence_transformer.encode(data, show_progress_bar=False)
        else:
            return None

    def process_data(self,
                     input_content: str,
                     generate_key_word=True,
                     generate_summary=True,
                     generate_summary_bert_vector=True):
        if not input_content:
            return None

        if not generate_summary:
            generate_summary_bert_vector = False

        res = []

        paragraphs = self.get_paragraph(input_content)

        for paragraph in paragraphs:
            res_dict = {'paragraph': paragraph}
            if generate_key_word:
                key_word = self.get_keyword(paragraph)
                res_dict['key_word'] = key_word

            if generate_summary:
                key_word_summary = self.get_keyword_summary(paragraph)

                if key_word_summary:
                    key_word_summary = ''.join(key_word_summary)

                res_dict['summary'] = key_word_summary

                if generate_summary_bert_vector:
                    if key_word_summary:
                        summary_embedding = self.get_summary_embedding(key_word_summary)
                        # res_dict['summary_bert_embedding'] = summary_embedding.tolist()
                        res_dict['summary_bert_embedding'] = summary_embedding[0].tolist()

                    else:
                        res_dict['summary_bert_embedding'] = None

            res.append(res_dict)

        return res

    def process_question_post(self,
                              question: str,
                              answers: list,
                              generate_key_word=True,
                              generate_summary=True,
                              generate_summary_bert_vector=True
                              ):

        res = []
        paragraphs = answers

        for paragraph in paragraphs:
            res_dict = {'paragraph': paragraph}
            content = paragraph + question
            if generate_key_word:
                key_word = self.get_keyword(content)
                res_dict['key_word'] = key_word

            if generate_summary:
                key_word_summary = self.get_keyword_summary(content)

                if key_word_summary:
                    key_word_summary = ''.join(key_word_summary)

                res_dict['summary'] = key_word_summary

                if generate_summary_bert_vector:
                    if key_word_summary:
                        summary_embedding = self.get_summary_embedding(key_word_summary)
                        res_dict['summary_bert_embedding'] = summary_embedding.tolist()
                    else:
                        res_dict['summary_bert_embedding'] = None

            res.append(res_dict)
        return res
