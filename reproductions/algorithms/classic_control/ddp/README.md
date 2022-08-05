# DDP (Differential Dynamic Programming) reproduction
This reproduction script runs the DDP (Differential Dynamic Programming) algorithm proposed by D. Mayne in the paper:
"A Second-order Gradient Method for Determining Optimal Trajectories of Non-linear Discrete-time Systems".
See also the below paper written by Y. Tassa et al.:
[Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf)
for the implementation of the algorithm.

NOTE: The algorithm is implemented in python and therefore very slow.

## How to run the reproduction script

To run the reproduction script do

```sh
$ python ddp_pendulum.py <options>
```

We recommend using --render option to check what is going on.

```sh
$ python ddp_pendulum.py --render
```

To check all available options type:

```sh
$ python ddp_pendulum.py --help
```