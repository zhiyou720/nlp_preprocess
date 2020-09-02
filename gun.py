# config.py
import gevent.monkey

gevent.monkey.patch_all()

# debug = True
bind = "0.0.0.0:10080"
pidfile = "logs/gunicorn.pid"
accesslog = "logs/access.logs"
errorlog = "logs/error.logs"

# 启动的进程数
workers = 4
worker_class = 'gevent'
x_forwarded_for_header = 'X-FORWARDED-FOR'
timeout = 600
