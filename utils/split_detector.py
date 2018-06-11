import copy
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from downloader import Downloader


class SplitObjectDetector(object):
    def __init__(self, model_url):
        download_request = Downloader.get(model_url)
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
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth=True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        with tf.Session(graph=input_graph, config=config):
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