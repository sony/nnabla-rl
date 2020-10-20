from nnabla_rl.hook import Hook

from nnabla_rl.utils.serializers import save_snapshot


class SaveSnapshotHook(Hook):
    def __init__(self, outdir, timing=1000):
        super(SaveSnapshotHook, self).__init__(timing=timing)
        self._outdir = outdir

    def on_hook_called(self, algorithm):
        save_snapshot(self._outdir, algorithm)
