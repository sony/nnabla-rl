# MPPI (Model Predictive Path Integral control) reproduction

This reproduction script runs the MPPI (Model Predictive Path Integral control) algorithm proposed by G. Williams, et al. in the paper:
    [Information theoretic MPC for model-based reinforcement learning](https://ieeexplore.ieee.org/document/7989202)

## How to run the reproduction script

To run the reproduction script do:

```sh
$ python mppi_pendulum.py <options>
```

The algorithm will try to learn the system dynamics and control the pendulum.
You will see the pendulum swinging up in less than 10 min.

We recommend using --render option to check what is going on.

```sh
$ python mppi_pendulum.py --render
```

You can use --use-known-dynamics option to use the true dynamics of the system instead of trained dynamics.

```sh
$ python mppi_pendulum.py --render --use-known-dynamics
```

To check all available options type:

```sh
$ python mppi_pendulum.py --help
```