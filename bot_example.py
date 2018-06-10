import time
import copy
import numpy as np
import cv2
from utils.category_index_coco import CATEGORY_INDEX as cateogry_index_coco
from utils.workers import SessionWorker
from utils.video_streamers import WebcamVideoStream
from utils.predictions_parser import parser
from utils.downloader import Downloader
import tensorflow as tf
from tensorflow.core.framework import graph_pb2

from multiprocessing import Process, Pipe

from easy_memmap import EasyMemmap, MultiImagesMemmap
import os
import datetime

MODEL_URL ="gs://vision-198622-production-models/object-detection/ssd_mobilenet_v1_coco_2017_11_17/model.pb" 

class FPS2:
    def __init__(self, interval):
        self._glob_start = None
        self._glob_end = None
        self._glob_numFrames = 0
        self._local_start = None
        self._local_numFrames = 0
        self._interval = interval
        self.curr_local_elapsed = None
        self.first = False

    def start(self):
        self._glob_start = datetime.datetime.now()
        self._local_start = self._glob_start
        return self

    def stop(self):
        self._glob_end = datetime.datetime.now()

    def update(self):
        self.first = True
        curr_time = datetime.datetime.now()
        self.curr_local_elapsed = (curr_time - self._local_start).total_seconds()
        self._glob_numFrames += 1
        self._local_numFrames += 1
        if self.curr_local_elapsed > self._interval:
          print("FPS: {}".format(self.fps_local()))
          self._local_numFrames = 0
          self._local_start = curr_time

    def elapsed(self):
        return (self._glob_end - self._glob_start).total_seconds()

    def fps(self):
        return self._glob_numFrames / self.elapsed()
    
    def fps_local(self):
        if self.first:
            return round(self._local_numFrames / self.curr_local_elapsed,1)
        else:
            return 0.0

class RealTimeObjectDetector(object):
    def __init__(self):
        download_request = Downloader.get(MODEL_URL)
        self.model_path = download_request.path
        self.call_model = True
        self.predictions = []

    def _node_name(self, n):
        if n.startswith("^"):
            return n[1:]
        else:
            return n.split(":")[0]

    def load_frozenmodel(self, model_path):
        print('> Loading frozen model into memory')
        num_classes = 90
        # load a frozen Model and split it into GPU and CPU graphs
        # Hardcoded for ssd_mobilenet
        input_graph = tf.Graph()
        with tf.Session(graph=input_graph):
            shape = 1917
            score = tf.placeholder(tf.float32, shape=(None, shape, num_classes), name="Postprocessor/convert_scores")
            expand = tf.placeholder(tf.float32, shape=(None, shape, 1, 4), name="Postprocessor/ExpandDims_1")
            for node in input_graph.as_graph_def().node:
                if node.name == "Postprocessor/convert_scores":
                    score_def = node
                if node.name == "Postprocessor/ExpandDims_1":
                    expand_def = node

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # model_path = 'frozen_inference_graph.pb' 
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                dest_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']

                edges = {}
                name_to_node_map = {}
                node_seq = {}
                seq = 0
                for node in od_graph_def.node:
                    n = self._node_name(node.name)
                    name_to_node_map[n] = node
                    edges[n] = [self._node_name(x) for x in node.input]
                    node_seq[n] = seq
                    seq += 1
                for d in dest_nodes:
                    assert d in name_to_node_map, "%s is not in graph" % d

                nodes_to_keep = set()
                next_to_visit = dest_nodes[:]
                
                while next_to_visit:
                    n = next_to_visit[0]
                    del next_to_visit[0]
                    if n in nodes_to_keep: continue
                    nodes_to_keep.add(n)
                    next_to_visit += edges[n]

                nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])
                nodes_to_remove = set()
                
                for n in node_seq:
                    if n in nodes_to_keep_list: continue
                    nodes_to_remove.add(n)
                nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])

                keep = graph_pb2.GraphDef()
                for n in nodes_to_keep_list:
                    keep.node.extend([copy.deepcopy(name_to_node_map[n])])

                remove = graph_pb2.GraphDef()
                remove.node.extend([score_def])
                remove.node.extend([expand_def])
                for n in nodes_to_remove_list:
                    remove.node.extend([copy.deepcopy(name_to_node_map[n])])

                with tf.device('/gpu:0'):
                    tf.import_graph_def(keep, name='')
                with tf.device('/cpu:0'):
                    tf.import_graph_def(remove, name='')

        return detection_graph, score, expand


    def detection(self):
        self.graph, self.score, self.expand = self.load_frozenmodel(self.model_path)
        print("Building Graph")
        detection_graph = self.graph
        score = self.score
        expand = self.expand
        # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False) # Log to true to debug
        config.gpu_options.allow_growth=True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        cur_frames = 0
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph,config=config) as sess:
                # Define Input and Ouput tensors
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
                # Threading
                gpu_worker = SessionWorker("GPU",detection_graph,config)
                cpu_worker = SessionWorker("CPU",detection_graph,config)
                gpu_opts = [score_out, expand_out]
                cpu_opts = [detection_boxes, detection_scores, detection_classes, num_detections]
                gpu_counter = 0
                cpu_counter = 0

                video_stream = MultiImagesMemmap(mode = "r", name = "main_stream", memmap_path = os.getenv("MEMMAP_PATH", "/tmp"))
                video_stream.wait_until_available() #initialize and find video data

                tick = time.time()
                print('Starting Detection')
                init_time = time.time()
                fps = FPS2(5).start()
                while True: # Always True
                    # if time.time() - init_time > 10:
                    #     self.call_model = not self.call_model
                    #     init_time = time.time()
                    if self.call_model:
                        gpu_worker.call_model = True
                        cpu_worker.call_model = True
                        if gpu_worker.is_sess_empty():
                            # read video frame, expand dimensions and convert to rgb
                            image = video_stream.read("C")
                            image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                            # put new queue
                            gpu_feeds = {image_tensor: image_expanded}
                            gpu_worker.put_sess_queue(gpu_opts,gpu_feeds)
        
                        g = gpu_worker.get_result_queue()
                        if g is None:
                            # gpu thread has no output queue. ok skip, let's check cpu thread.
                            gpu_counter += 1
                        else:
                            # gpu thread has output queue.
                            gpu_counter = 0
                            score,expand = g["results"][0],g["results"][1]

                            if cpu_worker.is_sess_empty():
                                # When cpu thread has no next queue, put new queue.
                                # else, drop gpu queue.
                                cpu_feeds = {score_in: score, expand_in: expand}
                                cpu_worker.put_sess_queue(cpu_opts,cpu_feeds,)

                        c = cpu_worker.get_result_queue()
                        if c is None:
                            # cpu thread has no output queue. ok, nothing to do. continue
                            cpu_counter += 1
                            time.sleep(0.005)
                            continue # If CPU RESULT has not been set yet, no fps update
                        else:
                            cpu_counter = 0
                            boxes, scores, classes, num = c["results"][0],c["results"][1],c["results"][2],c["results"][3]
                            # print("time: {}".format(time.time() - tick))
                            tick = time.time()
                            fps.update()
                        self.predictions = parser(num, boxes, scores, classes, image_shape=image.shape)
                        # print(self.predictions)
                        fps.update()
                    else:
                        gpu_worker.call_model = False
                        cpu_worker.call_model = False
                        time.sleep(0.1)

        # End everything
        gpu_worker.stop()
        cpu_worker.stop()


class RealTimeObjectDetectorProcess(Process):
    def __init__(self, **kwargs):
        super(RealTimeObjectDetectorProcess,self).__init__()
        self.object_detector = RealTimeObjectDetector()
        self.daemon = True

    def run(self):
        self.object_detector.detection()

def main():
    # object_detector = RealTimeObjectDetector()
    # object_detector.detection()
    object_process = RealTimeObjectDetectorProcess()
    object_process.start()
    object_process.join()

if __name__ == "__main__":
    main()