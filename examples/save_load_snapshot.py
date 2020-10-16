import nnabla_rl.algorithms as A
import nnabla_rl.environments as E
from nnabla_rl.utils import serializers


def main():
    params = A.DDPGParam(start_timesteps=200)
    env = E.DummyContinuous()
    ddpg = A.DDPG(env, params=params)

    outdir = './save_load_snapshot'

    # This actually saves the model and solver state right after the algorithm construction
    snapshot_dir = serializers.save_snapshot(outdir, ddpg)

    # This actually loads the model and solver state which is saved with the code above
    algorithm = serializers.load_snapshot(snapshot_dir)
    assert isinstance(algorithm, A.DDPG)


if __name__ == "__main__":
    main()
