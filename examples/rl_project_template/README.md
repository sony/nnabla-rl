# RL project template

This directory provides a minimum template for reinforcement learning (RL) projects.

## How to use

Start using this template by copying and pasting the entire directory and modifying the python files depending on your needs.
We created 3 main python files to get started.

- environment.py
- models.py
- training.py

See below descriptions for the usages of each file.

### environment.py

Sample implementation of an environment class.
Only the basic implementation is provided.
You will need to modify and add extra implementations to the file to 
properly to make the algorithm solve your problem.

### models.py

Sample implementation of DNN models to be learned by the RL algorithm.
You may also need to modify the models to get desired result.

### training.py

Main file that runs the training with RL algorithm.
See this file to understand the basics of how to implement the training process.

## How to run the script

Run the training.py script.
By default, it runs on cpu.

```sh
$ python training.py
```

To run on gpu, first, install nnabla-ext-cuda as follows.

```sh
# $ pip install nnabla-ext-cuda[cuda_version]
# Example: when you have installed CUDA-11.6 on your machine.
$ pip install nnabla-ext-cuda116
```

For the installation of nnabla-ext-cuda see also [here](https://github.com/sony/nnabla) and [here](https://github.com/sony/nnabla-ext-cuda).

Then, run the script by specifying the gpu id.

```sh
# This will run the script on gpu id 0.
$ python training.py --gpu=0
```