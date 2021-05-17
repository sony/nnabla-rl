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

class Writer(object):
    def __init__(self):
        pass

    def write_scalar(self, iteration_num, scalar):
        ''' Write scalar with your favorite tools

        Args:
            iteration_num (int): iteration number
            scalar (dict): scalar of the latest iteration state
        '''
        pass

    def write_histogram(self, iteration_num, histogram):
        ''' Write histogram with your favorite tools

        Args:
            iteration_num (int): iteration number
            histogram: histogram of the latest iteration state
        '''
        pass

    def write_image(self, iteration_num, image):
        ''' Write image with your favorite tools

        Args:
            iteration_num (int): iteration number
            image: image of the latest iteration state
        '''
        pass
