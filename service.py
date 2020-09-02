import configparser
import json
import logging
import logging.handlers
import logging.handlers
import os
import sys
import time
import traceback

from flask import Flask, make_response, request
from sentence_transformers import SentenceTransformer

import mongo_biz
import mongo_util
from keyword_main import KeyWordExtractor
from nlp_pipline import PipLine
from seg_main import Cutter

base_path = None
service_cfg = None
log_file = None
log_mode = None
log_time = None
app_host = None
app_port = None
PUNCS_PATH = None
SEG_MODEL_PATH = None
PUNC_MODEL_PATH = None
MAX_CUT_BATCH = None
MAX_SENTENCE_LENGTH = None

# key word parameters
SEED = None
TENCENT_EMBEDDING_PATH = None
USER_DICT_PATH = None
CONSIDER_TAGS_PATH = None
STOP_WORDS_PATH = None
SIF_WEIGHT_PATH = None
BERT_MODEL_NAME = None
SIF_EMBEDDING_PATH = None
pip_line = None

mongo_host = None
mongo_port = None
mongo_username = None
mongo_password = None
mongo_schema = None
mongo_helper = None


# Load configuration
def load_conf():
    global base_path
    global service_cfg
    global log_file
    global log_mode
    global log_time
    global app_host
    global app_port
    global PUNCS_PATH
    global SEG_MODEL_PATH
    global PUNC_MODEL_PATH
    global MAX_CUT_BATCH
    global MAX_SENTENCE_LENGTH
    global SEED
    global TENCENT_EMBEDDING_PATH
    global USER_DICT_PATH
    global CONSIDER_TAGS_PATH
    global STOP_WORDS_PATH
    global SIF_WEIGHT_PATH
    global BERT_MODEL_NAME
    global SIF_EMBEDDING_PATH
    global mongo_host
    global mongo_port
    global mongo_username
    global mongo_password
    global mongo_schema
    try:
        base_path = os.path.dirname(os.path.realpath(__file__))
        service_cfg = configparser.ConfigParser()
        service_cfg.read('%s/conf/service.conf' % base_path)
        app_host = service_cfg.get('base', 'host')
        app_port = int(service_cfg.get('base', 'port'))
        log_file = '%s/logs/%s' % (base_path, service_cfg.get('base', 'log_file'))
        log_mode = service_cfg.get('base', 'log_mode')
        log_time = int(service_cfg.get('base', 'log_time'))

        PUNCS_PATH = service_cfg.get('model', 'PUNCS_PATH')
        SEG_MODEL_PATH = service_cfg.get('model', 'SEG_MODEL_PATH')
        PUNC_MODEL_PATH = service_cfg.get('model', 'PUNC_MODEL_PATH')
        MAX_CUT_BATCH = int(service_cfg.get('model', 'MAX_CUT_BATCH'))
        MAX_SENTENCE_LENGTH = int(service_cfg.get('model', 'MAX_SENTENCE_LENGTH'))

        # key word parameters
        SEED = int(service_cfg.get('model', 'SEED'))
        TENCENT_EMBEDDING_PATH = service_cfg.get('model', 'TENCENT_EMBEDDING_PATH')
        USER_DICT_PATH = service_cfg.get('model', 'USER_DICT_PATH')
        CONSIDER_TAGS_PATH = service_cfg.get('model', 'CONSIDER_TAGS_PATH')
        STOP_WORDS_PATH = service_cfg.get('model', 'STOP_WORDS_PATH')
        SIF_WEIGHT_PATH = service_cfg.get('model', 'SIF_WEIGHT_PATH')
        BERT_MODEL_NAME = service_cfg.get('model', 'BERT_MODEL_NAME')
        SIF_EMBEDDING_PATH = service_cfg.get('model', 'SIF_EMBEDDING_PATH')

        mongo_host = service_cfg.get('mongodb', 'host')
        mongo_port = int(service_cfg.get('mongodb', 'port'))
        mongo_username = service_cfg.get('mongodb', 'username')
        mongo_password = service_cfg.get('mongodb', 'password')
        mongo_schema = service_cfg.get('mongodb', 'schema')
        print('Load configuration success')
        return True
    except Exception as expt:
        traceback.print_exc()
        print('Load configuration failure')
        return False


# Initialize logs
def init_log():
    global log_file
    global log_mode
    global log_time
    try:
        logger = logging.getLogger()
        if log_mode == 'debug':
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        fileHandler = logging.handlers.TimedRotatingFileHandler(log_file, when='d', interval=1, backupCount=log_time)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.formatter = formatter
        logger.addHandler(consoleHandler)
        print('Initialize logging success')
        return True
    except Exception as expt:
        print('Initialize logging failure')
        traceback.print_exc()
        return False


# Initialize system
def init_sys():
    global pip_line
    global mongo_host
    global mongo_port
    global mongo_username
    global mongo_password
    global mongo_schema
    global mongo_helper
    try:
        CUTTER = Cutter(puncs_path=PUNCS_PATH,
                        seg_model_path=SEG_MODEL_PATH,
                        punc_model_path=PUNC_MODEL_PATH,
                        max_cut_batch=MAX_CUT_BATCH,
                        max_sentence_length=MAX_SENTENCE_LENGTH)

        SENT_TRANS_MODEL = SentenceTransformer(BERT_MODEL_NAME)

        KEY_WORD_EXTRACTOR = KeyWordExtractor(user_word_path=USER_DICT_PATH,
                                              puncs_path=PUNCS_PATH,
                                              considered_tags_path=CONSIDER_TAGS_PATH,
                                              stop_word_path=STOP_WORDS_PATH,
                                              sif_embedding_path=SIF_EMBEDDING_PATH,
                                              sif_weight_path=SIF_WEIGHT_PATH,
                                              tencent_embedding_path=TENCENT_EMBEDDING_PATH,
                                              random_seed=SEED)

        SEG_PROCESSOR = CUTTER.seg_processor
        KEY_WORD_PROCESSOR = KEY_WORD_EXTRACTOR.key_word_func

        KEY_WORD_TOPICS = 3
        KEY_WORD_TOP_N = 2

        pip_line = PipLine(seg_processor=SEG_PROCESSOR,
                           sentence_transformer=SENT_TRANS_MODEL,
                           key_word_func=KEY_WORD_PROCESSOR,
                           keyword_max_k=KEY_WORD_TOPICS, keyword_top_n=KEY_WORD_TOP_N
                           )
        mongo_helper = mongo_util.MongodbHelper(mongo_host, mongo_port, mongo_schema, mongo_username, mongo_password)
        logging.info('Initialize system success')
        return True
    except Exception as expt:
        mysql_helper = None
        traceback.print_exc()
        logging.exception(expt)
        logging.error('Initialize system failure')
        return False


app = Flask(__name__)

# Load configuration & environment
if not load_conf():
    sys.exit(1)
# Initialize logging
if not init_log():
    sys.exit(2)
if not init_sys():
    sys.exit(3)


@app.errorhandler(405)
def error_405_handler(error):
    if error:
        logging.error(error)
    data = {'result': False, 'msg': 'Method not support error'}
    resp = make_response(json.dumps(data, ensure_ascii=False), 405, {'Content-Type': 'application/json'})
    return resp


@app.errorhandler(404)
def error_404_handler(error):
    if error:
        logging.error(error)
    data = {'result': False, 'msg': 'Not found error'}
    resp = make_response(json.dumps(data, ensure_ascii=False), 404, {'Content-Type': 'application/json'})
    return resp


@app.errorhandler(403)
def error_403_handler(error):
    if error:
        logging.error(error)
    data = {'result': False, 'msg': 'Forbidden error'}
    resp = make_response(json.dumps(data, ensure_ascii=False), 403, {'Content-Type': 'application/json'})
    return resp


@app.errorhandler(500)
def error_500_handler(error):
    if error:
        logging.error(error)
    data = {'result': False, 'msg': 'Server error'}
    resp = make_response(json.dumps(data, ensure_ascii=False), 500, {'Content-Type': 'application/json'})
    return resp


@app.route('/heartbeat', methods=['GET'])
def heartbeat_servlet():
    try:
        heartbeat = {}
        tstamp = int(time.time())
        heartbeat['heartbeat'] = 1
        heartbeat['timestamp'] = tstamp
        return json.dumps(heartbeat, ensure_ascii=False), {'Content-Type': 'application/json'}
    except Exception as expt:
        traceback.print_exc()
        logging.exception(expt)
        logging.error('Heartbeat exception')
        heartbeat = {}
        tstamp = int(time.time())
        heartbeat['heartbeat'] = 0
        heartbeat['timestamp'] = tstamp
        resp = make_response(json.dumps(heartbeat, ensure_ascii=False), 500, {'Content-Type': 'application/json'})
        return resp


@app.route('/seg', methods=['POST'])
def segmenting():
    global processor
    try:
        params = json.loads(request.data)
        html_content = params['content']
        logging.debug(html_content)
        if html_content:
            data = pip_line.process_data(html_content, generate_key_word=True, generate_summary=True,
                                         generate_summary_bert_vector=True)
        data = {'result': True, 'segments': data}
        return json.dumps(data, ensure_ascii=False), {'Content-Type': 'application/json'}
    except Exception as expt:
        traceback.print_exc()
        logging.exception(expt)
        return return_error(expt)


@app.route('/seg/db', methods=['POST'])
def segmenting_db():
    global processor
    global mongo_helper
    try:
        params = json.loads(request.data)
        core_id = int(params['core_id'])
        object_id = params['object_id']
        article_type = int(params['article_type'])
        insert_ids = []
        if article_type == 3:
            question = params['title']
            answers = params['answers']
            if question and answers:
                process_data = pip_line.process_question_post(question=question,
                                                              answers=answers,
                                                              generate_key_word=True,
                                                              generate_summary=True,
                                                              generate_summary_bert_vector=True)
                if process_data:
                    new_data = list(d for d in process_data if d['key_word'])
                    if new_data:
                        insert_ids = mongo_biz.insert_segments(mongo_helper, object_id, core_id, article_type, new_data)
        else:
            html_content = params['content']
            if html_content:
                logging.debug(html_content)
                process_data = pip_line.process_data(html_content, generate_key_word=True, generate_summary=True,
                                                     generate_summary_bert_vector=True)
                if process_data:
                    new_data = list(d for d in process_data if d['key_word'])
                    if new_data:
                        insert_ids = mongo_biz.insert_segments(mongo_helper, object_id, core_id, article_type, new_data)

        data = {'result': True, 'ids': insert_ids}
        return json.dumps(data, ensure_ascii=False), {'Content-Type': 'application/json'}
    except Exception as expt:
        logging.info(params)
        traceback.print_exc()
        logging.exception(expt)
        return return_error(expt)


# Handler
def return_error(expt):
    if expt and expt.message:
        data = {'result': False, 'msg': expt.message}
        resp = make_response(json.dumps(data, ensure_ascii=False), 500, {'Content-Type': 'application/json'})
        return resp
    else:
        raise Exception('Return error exception')


# Main function
if __name__ == '__main__':
    # Loop server
    logging.info('Run server')
    if log_mode == 'debug':
        app.run(host=app_host, port=int(app_port), debug=True, threaded=True)
    else:
        app.run(host=app_host, port=int(app_port), debug=False, threaded=True)
