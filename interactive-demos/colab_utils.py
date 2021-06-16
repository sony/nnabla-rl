# Copyright 2021 Sony Corporation.
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

from IPython import display
from matplotlib import pyplot as plt
from pyvirtualdisplay import Display

from nnabla_rl.hook import Hook


class RenderHook(Hook):
    def __init__(self, env):
        super(RenderHook, self).__init__(timing=1)
        '''
        Hook to render environment during training.
        '''
        self._env = env
        self._display = Display()
        self._display.start()
        self._image = None
        self._iteration_num = 0

    def on_hook_called(self, algorithm):
        display.clear_output(wait=True)
        if self._image is None:
            self._image = plt.imshow(self._env.render('rgb_array'))
        else:
            self._image.set_data(self._env.render('rgb_array'))
        plt.suptitle(f"iteration num : {self._iteration_num}")
        self._iteration_num += 1
        plt.axis('off')
        display.display(plt.gcf())

    def reset(self):
        self._image = None
        self._iteration_num = 0
        self._display = Display()
        self._display.start()
