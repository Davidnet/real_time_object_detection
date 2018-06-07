import tfinterface as ti

class Downloader(ti.estimator.getters.FileGetter):
    def __init__(self, frozen_graph_path):
        self._path = frozen_graph_path

    @property
    def path(self):
        """Get the current path"""
        return self._path
