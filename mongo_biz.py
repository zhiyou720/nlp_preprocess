#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import traceback


def insert_segments(mongo_helper, object_id, core_id, article_type, data):
    try:
        with mongo_helper.operDb() as db:
            collection = db['article_segments']
            for index, dict_doc in enumerate(data):
                logging.debug(dict_doc)
                dict_doc['article_id'] = object_id
                dict_doc['article_core_id'] = core_id
                dict_doc['article_type'] = article_type
                dict_doc['seq'] = index
            results = collection.insert_many(data)
            if results:
                insert_ids = list(map(lambda x: x.__str__(), results.inserted_ids))
                logging.info('Insert segments success <%s>' % insert_ids)
                return insert_ids
            else:
                logging.error('Insert segments failure')
                raise Exception('Insert segments failure')
    except Exception as expt:
        traceback.print_exc()
        logging.exception(expt)
        logging.error('Insert segments error')
        raise Exception('Insert segments error')
