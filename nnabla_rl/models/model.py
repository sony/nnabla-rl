import nnabla as nn

import pathlib


class Model(object):
    def __init__(self, scope_name):
        self._scope_name = scope_name

    @property
    def scope_name(self):
        return self._scope_name

    def get_parameters(self):
        with nn.parameter_scope(self.scope_name):
            return nn.get_parameters()

    def save_parameters(self, filepath):
        if isinstance(filepath, pathlib.Path):
            filepath = str(filepath)
        with nn.parameter_scope(self.scope_name):
            nn.save_parameters(path=filepath)

    def load_parameters(self, filepath):
        if isinstance(filepath, pathlib.Path):
            filepath = str(filepath)
        with nn.parameter_scope(self.scope_name):
            nn.load_parameters(path=filepath)
