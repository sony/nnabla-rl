# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nnabla.experimental.viewers as V
from nnabla_rl.hook import Hook
from nnabla_rl.logger import logger


def print_network(x):
    def accept_nnabla_func(nnabla_func):
        print("==========")
        print(nnabla_func.info.type_name)
        print(nnabla_func.inputs)
        print(nnabla_func.outputs)
        print(nnabla_func.info.args)
    x.visit(accept_nnabla_func)


def view_graph(x, verbose=False):
    graph = V.SimpleGraph(verbose=verbose)
    graph.view(x)


def save_graph(x, file_path, verbose=False):
    graph = V.SimpleGraph(verbose=verbose)
    graph.save(x, file_path)


def count_parameter_number(parameters):
    '''
    Args:
        parameters (dict): parameters in dictionary form
    Returns:
        parameter_number (int): parameter number
    '''
    parameter_number = 0
    for parameter in parameters.values():
        parameter_number += parameter.size
    return parameter_number


try:
    from pympler import muppy, summary

    class PrintMemoryDumpHook(Hook):
        def __init__(self, timing):
            super(PrintMemoryDumpHook, self).__init__(timing=timing)

        def on_hook_called(self, _):
            all_objects = muppy.get_objects()
            summarized = summary.summarize(all_objects)
            self.print_summary(summarized)

        def print_summary(self, rows, limit=30, sort='size', order='descending'):
            for line in summary.format_(rows, limit=limit, sort=sort, order=order):
                logger.debug(line)
except ModuleNotFoundError:
    pass
