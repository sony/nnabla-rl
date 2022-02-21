# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022 Sony Group Corporation.
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

import pathlib
from typing import Optional, Union

import nnabla as nn
import nnabla.experimental.viewers as V
import nnabla.solvers as S
from nnabla.utils.profiler import GraphProfiler, GraphProfilerCsvWriter
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


def profile_graph(
    output_variable: nn.Variable,
    csv_file_path: Union[str, pathlib.Path],
    solver: Optional[S.Solver] = None,
    ext_name: str = 'cudnn',
    device_id: int = 0,
    n_run: int = 1000,
) -> None:
    '''Profile computational graph. Print the profile result to console and save it to the csv_file_path.

    Args:
        output_variable (nn.Variable): output variable of the graph.
        csv_file_path (Union[str, pathlib.Path]): csv file path of the profile results.
        solver (Optional[S.Solver]): nnabla solver, if this parameter is not None,
            nnabla Profiler also measures updating the parameter, defaults to None.
        ext_name (str): extention name, defaults to cudnn.
        device_id (int): device id, defaults to 0.
        n_run: (int): number of runs, defaults to 1000.

    Examples:
        >>> import nnabla as nn
        >>> import nnabla.functions as NF
        >>> from nnabla_rl.utils.debugging import profile_graph
        >>> x = nn.Variable([1, 1])
        >>> y = NF.relu(x)
        >>> output_file_path = "sample.csv"
        >>> profile_graph(y, output_file_path)
        # The profile result is shown in the console and the result is saved as sample.csv.
    '''
    B = GraphProfiler(output_variable, solver=solver, device_id=device_id, ext_name=ext_name, n_run=n_run)
    B.run()
    B.print_result()
    with open(csv_file_path, "w") as f:
        writer = GraphProfilerCsvWriter(B, file=f)
        writer.write()


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
