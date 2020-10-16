# Using Writer with Latest Iteration State and Iteration State Hook

nnabla_rl provides useful [IterationStateHook](iteration_state_hook.py) to debug and record the training process.

You can easily save/write the latest iteration state(See below) using your favorite writer tools such as nnabla.Monitor,  TensorboardX, and Trains.

At first, we will describe what latest iteration state is.

Latest iteration state is a dictionary with following keys

## Scalar

This key contains scalar outputs such as training loss. It should be noted that each algorithm will save different scalars. 

## Histogram

This key contains any distributional values such as weights and biases of a network. It should be noted that each algorithm will save different histograms. 

## Image

This key contains an image that follows (batch_size, channel, height, width) order. It should be noted that each algorithm will save different images.

If you customize the latest iteration state to fit your situation, please see 'Implementing a latest iteration state' section.

Next, we will describe how to use a writer with latest iteration state hook and latest iteration state.

In nnabla_rl, we provide Writer class(See [here](../writer.py)) so please inherit that.

This is an example of a writer that saves scalar of the latest iteration state with Monitor of nnabla.

```py
# example_writer.py
from nnabla_rl.utils.files import create_dir_if_not_exist
from nnabla_rl.writer import Writer

from nnabla.monitor import Monitor, MonitorSeries


class MyScalarWriter(Writer):
    def __init__(self, outdir):
        self._outdir = os.path.join(outdir, 'writer')
        create_dir_if_not_exist(outdir=self._outdir)
        self._monitor = Monitor(self._outdir)
        self._monitor_series = None
        super().__init__()

    def write_scalar(self, iteration_num, scalar):
        if self._monitor_series is None:
            self._create_monitor_series(scalar.keys())

        for writer, value in zip(self._monitor_series, scalar.values()):
            writer.add(iteration_num, value)

    def _create_monitor_series(self, names):
        self._monitor_series = []
        for name in names:
            self._monitor_series.append(MonitorSeries(
                name, self._monitor, interval=1, verbose=False))
```

Next, you add the iteration state hook with the writer to the hooks.

Then the writer is called within the iteration state hook at every timing you set.

If you do not pass the writer to iteration state hook, the iteration state hook only prints the scalar of latest iteration state.

```py
writer = MyScalarWriter(outdir)
iteration_state_hook = IterationStateHook(writer=writer)

train_env = build_env()
algorithm = A.MyAlgorithm(train_env, hooks=[
                           iteration_state_hook])
algorithm.train(train_env)
```

# Implementing a latest iteration state

Because the latest iteration states contains all-important states in all implemented algorithm of nnabla_rl, you don't have to care what kind of latest iteration state is saved. 

However, in the case that you implement an original algorithm, please override the _latest_iteration_state method of Algorithm class and you can save any parameters you want.

```py
# myalgorithm.py
Class MyAlgorithm(Algorithm):
    def __init__(...):
    ###
    ...
    ###

    def latest_iteration_state(self):  # Please override this method.
        latest_iteration_state = {}
        latest_iteration_state['scalar'] = {}
        latest_iteration_state['histogram'] = {}
        latest_iteration_state['image'] = {}

        # self.loss is nnabla.Variable
        latest_iteration_state['scalar']['loss'] = self.loss.d.flatten()  
        
        # self.network.get_parameters() returns network parameters as dictionary
        latest_iteration_state['histogram'].update(self.network.get_parameters())

        # shape of self.images is (batch_size, channel, height, width)
        latest_iteration_state['image'] = self.images.d
        
        return latest_iteration_state
```