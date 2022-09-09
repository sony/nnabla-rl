# Copyright 2022 Sony Group Corporation.
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

import os

from nnabla.utils.save import save
from nnabla_rl.hook import Hook
from nnabla_rl.logger import logger


class TrainingGraphHook(Hook):
    '''
    Hook to save training computational graphs as nntxt.

    Args:
        outdir (str): Output directory.
        name (str): Name of nntxt file.
    '''

    def __init__(self, outdir, name="training"):
        super(TrainingGraphHook, self).__init__(timing=1)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self._outdir = outdir
        self._name = name
        self._saved = False

    def on_hook_called(self, algorithm):
        # skip after the first call
        if self._saved:
            return

        networks = []
        executors = []

        for name, trainer in algorithm.trainers.items():
            input_variables = trainer.training_variables.get_variables()
            output_variables = trainer.loss_variables

            # remove unreferenced input variables
            filtered_input_variables = {}
            for variable_name, variable in input_variables.items():
                if variable.function_references:
                    filtered_input_variables[variable_name] = variable

            network = {
                "name": name,
                "batch_size": 1,
                "outputs": output_variables,
                "names": filtered_input_variables,
            }
            networks.append(network)
            executor = {
                "name": name,
                "network": name,
                "data": list(filtered_input_variables.keys()),
                "output": list(output_variables.keys()),
            }
            executors.append(executor)

        path = os.path.join(self._outdir, f"{self._name}.nntxt")
        contents = {"networks": networks, "executors": executors}

        save(path, contents)
        logger.info(f'Training computational graphs have been saved to {path}.')

        self._saved = True
