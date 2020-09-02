#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import traceback
from contextlib import contextmanager

from pymongo import MongoClient


# Mongodb helper
class MongodbHelper:
    def __init__(self, host, port, schema, username, password):
        try:
            self.host = host
            self.port = port
            self.schema = schema
            self.username = username
            self.password = password
            uri = 'mongodb://%s:%s@%s:%d/%s' % (self.username, self.password, self.host, self.port, self.schema)
            self.client = MongoClient(uri, connect=False, maxPoolSize=200)
            logging.info('Init helper success')
        except Exception as expt:
            traceback.print_exc()
            logging.exception(expt)
            logging.error('Init helper failure')

    @contextmanager
    def operDb(self):
        try:
            db = self.client[self.schema]
            # db.authenticate(self.username, self.password)
            yield db
        except Exception as expt:
            traceback.print_exc()
            logging.exception(expt)
            logging.error('Oper mongodb error')
            raise Exception('Oper mongodb error')
