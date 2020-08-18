# Tokenizer 中的标点必须与标点符号预测模型中的需要预测的标点符号一致
# SegAndPunc 类中的标点必须与标点符号预测模型中的需要预测的标点符号一致
# 不计算组合标签
# TODO: 使用TODO标记 现有标点符号预测模型还需要更新：1. 使用更多更好的数据。 2. 使用括号类标签使用CRF提高准确率 3. 组合标签不足

import re
import time
import torch
import json
import ijson
import logging
import argparse
from model import Ner
from pyparsing import oneOf
from tokenizer import tokenize
from dataio import load_txt_data
from h5_preprocessor import process_html_tag
from format_processed_data import token2sent

logging.basicConfig(filename='logger.log', level=logging.INFO)

DEVICE = torch.device('cuda')
PUNC_TAGS = [x.split('\t')[1] for x in load_txt_data('model/config/punctuation.dat')]
# TODO: 不再需要第一列
print('LOAD MODEL')
SEG_MODEL = Ner('model/seg_model')
PUNC_MODEL = Ner('model/punc_model')
print('FINISHED LOAD')


class SegAndPunc:
    def __init__(self, seg_model, punc_model, punc_tags,
                 max_cut_batch=502,
                 max_sent_len=130
                 ):

        self.seg_model = seg_model
        self.punc_model = punc_model
        self.punc_tags = punc_tags
        self.max_cut_batch = max_cut_batch
        self.max_sent_len = max_sent_len

    @staticmethod
    def format_predict_data(data):
        res = []
        for _item in data:
            res.append(_item['word'])
            if _item['tag'] != 'word':
                res.append(_item['tag'])
        return res

    def remove_puncs(self, data):
        # TODO: 与被预测标点保持一致
        puncs = ['\\' + x for x in self.punc_tags]
        for item in puncs:
            data = re.sub(item, '', data)
        return data

    def tag_punc(self, sentence):
        return self.format_predict_data(self.punc_model.predict('b ' + ' '.join(sentence)))[1:]

    @staticmethod
    def mark_conflict_symbol(data):
        data = re.sub('#', '@shapmask@', data)
        data = re.sub('\$', '@dollarmask@', data)
        return data

    @staticmethod
    def re_mark_conflict_symbol(data):
        """
        :param data: list[list]
        :return:
        """
        for i in range(len(data)):
            for j in range(len(data)):
                data[i][j] = re.sub('@dollarmask@', '$', data[i])
                data[i][j] = re.sub('@shapmask@', '#', data[i])
        return data

    @staticmethod
    def split_paragraph(data):
        """
        预测分段符之后进行分段
        :param data: []
        :return: [[],[],[]]
        :rtype:list
        """
        res = []
        i = 0
        part = []

        if '$' and '#' not in data:
            return [data]
        while i < len(data):
            if data[i] == '#':
                part.append(i - 1)

            elif data[i] == '$':
                part.append(i)

            if len(part) == 2:
                slice_part = data[part[0]:part[1]]
                # slice_part = re.sub('[#$]', '', slice_part)
                while '#' in slice_part:
                    slice_part.remove('#')
                while '$' in slice_part:
                    slice_part.remove('$')
                res.append(slice_part)
                part.pop(0)
            i += 1
        if part[0] != len(data) - 1:
            slice_part = data[part[0]:]
            # slice_part = re.sub('[#$]', '', slice_part)
            while '#' in slice_part:
                slice_part.remove('#')
            while '$' in slice_part:
                slice_part.remove('$')
            res.append(slice_part)
        res = [x for x in res if x]
        return res

    def seg_data(self, data):
        return self.format_predict_data(self.seg_model.predict(data))

    def punc_help(self, data):
        """
        :param data:  [chr, chr, chr]
        :return: [[chr, chr], [chr, chr]]
        """
        _ = self.tag_punc(data)
        _ = ' '.join(_)
        seg_punc = oneOf(list('。?？！!'))
        _ = seg_punc.split(_)
        _ = [self.remove_puncs(x).split() for x in _]
        _ = [x for x in _ if x]
        split_index = int(len(_) / 2)
        part1 = [y for x in _[:split_index] for y in x]
        part2 = [y for x in _[split_index:] for y in x]
        if part1:
            return [part1, part2]
        else:
            return [part2]

    def seg_loop(self, data):
        """
        对小于512长度的文本逐句划分
        :param data:
        :return:
        """
        res = []
        while data:
            data = self.seg_paragraph(data)
            if len(data) > 1:
                res.append(data[0])
                data = [y for x in data[1:] for y in x]
            else:
                res.append(data[0])
                data = None
        return res

    def seg_paragraph(self, data):
        """
        :param data: list
        :return: list[list]
        """
        replace_flag = False
        data = ' '.join(data)

        if '#' in data or '$' in data:
            replace_flag = True

        if replace_flag:
            data = self.mark_conflict_symbol(data)

        data = self.seg_data(data)
        data = self.split_paragraph(data)

        if len(data[0]) > self.max_sent_len:
            cur = data[0]
            rest = data[1:]
            tmp = self.seg_data(' '.join(cur))
            tmp = self.split_paragraph(tmp)

            if len(tmp[0]) < self.max_sent_len:
                data = tmp + rest
            else:
                data = self.punc_help(tmp[0])
                if len(tmp[1:]):
                    tmp = tmp[1:] + rest
                else:
                    tmp = rest
                data = data + tmp

                if len(data[0]) > self.max_sent_len:
                    data = self.punc_help(data[0]) + data[1:]

                if len(data[0]) > self.max_sent_len:
                    # TODO: 强制
                    data = [data[0][:128]] + [data[0][128:]] + data[1:]
        if '!' == data[-1][-1]:
            data[-1] = data[-1][:-1]

        if replace_flag:
            data = self.re_mark_conflict_symbol(data)

        return data

    def cut_oov_data(self, data):
        if len(data) > self.max_cut_batch:
            cur = data[:self.max_cut_batch]
            rest = data[self.max_cut_batch:]
        else:
            cur = data
            rest = None
        return cur, rest

    def seg_func(self, data):
        """

        :param data:[chr, chr, chr]
        :return:
        """
        if not data:
            return None
        res = []
        data_len = len(data)

        if self.max_sent_len < data_len:
            batch, rest_data = self.cut_oov_data(data)
            if rest_data:
                while rest_data:
                    cur_data_seg = self.seg_paragraph(batch)

                    res.append(cur_data_seg[0])
                    tem_rest = [y for x in cur_data_seg[1:] for y in x]

                    rest_data = tem_rest + rest_data
                    batch, rest_data = self.cut_oov_data(rest_data)

                last_seg = self.seg_loop(batch)
                res = res + last_seg
            else:
                res = self.seg_loop(batch)
        else:
            res = [data]
        return res

    def punc_func(self, data):
        """
        PUNC PREDICT
        :param data: [[], []]
        :return:
        """
        if not data:
            return None
        content = []
        for sentence in data:
            sent = self.tag_punc(sentence)
            content.append(sent)
        return content


def process(data, processor, log=True):
    """

    :param processor: 实例化 SegAndPunc 类
    :param data: DB 中 content 字段
    :type data: str
    :return:[[chr chr chr], [...]]; '\n'.join(x)
    """
    '''Process HTML Data'''
    data = process_html_tag(data)

    if log:
        logging.info('Process HTML data Done!')
        for item in data:
            logging.info(item)

    data = ''.join(data)

    '''Tokenize Data'''
    data = tokenize(data)

    data = processor.seg_func(data)

    data = processor.punc_func(data)

    data = token2sent(data)

    if log:
        logging.info('Process Seg and Punc data Done!')
        for item in data:
            logging.info(item)

    return data


def main(args):
    processor = SegAndPunc(seg_model=SEG_MODEL, punc_model=PUNC_MODEL, punc_tags=PUNC_TAGS,
                           max_cut_batch=args.max_cut_len,
                           max_sent_len=args.max_sent_len)
    with open(args.data_dir, 'r', encoding='utf-8', errors='ignore') as f:
        data = ijson.items(f, 'RECORDS.item')
        new_data = []
        i = 0
        k = 3
        start = time.time()
        while True:
            try:
                if i < 25200:
                    _data = data.__next__()
                    i += 1
                    continue
                loop_time_start = time.time()
                _data = data.__next__()

                object_id = _data['object_id']
                content = _data['content']
                title = _data['title']

                format_content = process(content, processor=processor)
                exit()
                tmp = {'object_id': object_id, 'title': title, 'format_content': format_content}
                new_data.append(tmp)

                i += 1
                loop_time_end = time.time()
                print('processed {}th data: time used: {}s'.format(i,
                                                                   round(loop_time_end - loop_time_start,
                                                                         2)))
                print(round(loop_time_end - start, 2))
                if i % 100 == 0:
                    new_data = {'DATA': new_data}
                    res_path = args.res_dir.format(str(k))
                    with open(res_path, "w", encoding='utf-8') as save_file:
                        json.dump(new_data, save_file, indent=4, ensure_ascii=False)

                    print('save data to {}'.format(res_path))
                    k += 1
                    new_data = []

            except StopIteration as e:
                if new_data:
                    new_data = {'DATA': new_data}
                    res_path = args.res_dir.format(str(k))
                    with open(res_path, "w") as save_file:
                        json.dump(new_data, save_file)

                print("数据读取完成")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/db_data.1.json', type=str)
    parser.add_argument("--res_dir", default='res/db_data_res.{}.json', type=str)
    parser.add_argument("--punc_conf", default='src/config/punctuation.dat', type=str)
    parser.add_argument("--seg_conf", default='src/config/segmentation.dat', type=str)
    parser.add_argument("--max_sent_len", default=130, type=int, help="maximum sentence length")
    parser.add_argument("--max_cut_len", default=502, type=int, help="maximum cut length")
    _args = parser.parse_args()
    main(_args)
