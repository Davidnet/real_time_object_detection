import time
import threading 

import sys
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY2:
    import Queue
elif PY3:
    import queue as Queue

import tensorflow as tf

class SessionWorker():
# from https://github.com/naisy/realtime_object_detection/blob/master/lib/session_worker.py
# TensorFlow Session Thread
#
# usage:
# before:
#     results = sess.run([opt1,opt2],feed_dict={input_x:x,input_y:y})
# after:
#     opts = [opt1,opt2]
#     feeds = {input_x:x,input_y:y}
#     woker = SessionWorker("TAG",graph,config)
#     worker.put_sess_queue(opts,feeds)
#     q = worker.get_result_queue()
#     if q is None:
#         continue
#     results = q['results']
#     extras = q['extras']
#
# extras: None or frame image data for draw. GPU detection thread doesn't wait result. Therefore, keep frame image data if you want to draw detection result boxes on image.
#
    def __init__(self,tag,graph,config):
        self.lock = threading.Lock()
        self.sess_queue = Queue.Queue()
        self.result_queue = Queue.Queue()
        self.tag = tag
        t = threading.Thread(target=self.execution,args=(graph,config))
        t.setDaemon(True)
        t.start()
        return

    def execution(self,graph,config):
        self.is_thread_running = True
        try:
            with tf.Session(graph=graph,config=config) as sess:
                while self.is_thread_running:
                        while not self.sess_queue.empty():
                            q = self.sess_queue.get(block=False)
                            opts = q["opts"]
                            feeds= q["feeds"]
                            extras= q["extras"]
                            if feeds is None:
                                results = sess.run(opts)
                            else:
                                results = sess.run(opts,feed_dict=feeds)
                            self.result_queue.put({"results":results,"extras":extras})
                            self.sess_queue.task_done()
                        time.sleep(0.005)
        except:
            import traceback
            traceback.print_exc()
        self.stop()
        return

    def is_sess_empty(self):
        if self.sess_queue.empty():
            return True
        else:
            return False

    def put_sess_queue(self,opts,feeds=None,extras=None):
        self.sess_queue.put({"opts":opts,"feeds":feeds,"extras":extras})
        return

    def is_result_empty(self):
        if self.result_queue.empty():
            return True
        else:
            return False

    def get_result_queue(self):
        result = None
        if not self.result_queue.empty():
            result = self.result_queue.get(block=False)
            self.result_queue.task_done()
        return result

    def stop(self):
        self.is_thread_running=False
        with self.lock:
            while not self.sess_queue.empty():
                q = self.sess_queue.get(block=False)
                self.sess_queue.task_done()
        return