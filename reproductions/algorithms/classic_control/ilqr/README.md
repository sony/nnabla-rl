# iLQR (iterative Linear Quadratic Regulator) reproduction

This reproduction script runs the iLQR (iterative LQR) algorithm proposed by Y. Tassa, et al. in the paper:
    [Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf)

NOTE: The algorithm is implemented in python and therefore very slow.

## How to run the reproduction script

To run the reproduction script for acrobot do

```sh
$ python ilqr_acrobot.py <options>
```

for pendulum do

```sh
$ python ilqr_pendulum.py <options>
```

We recommend using --render option to check what is going on.

```sh
$ python ilqr_acrobot.py --render
```

```sh
$ python ilqr_pendulum.py --render
```

To check all available options type:

```sh
$ python ilqr_acrobot.py --help
```

or

```sh
$ python ilqr_pendulum.py --help
```