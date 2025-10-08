def test_skeleton_modules_present():
    import importlib

    # Modules that should exist at this stage
    for mod in [
        "irl",
        "irl.cfg",
        "irl.envs",
        "irl.models",
        "irl.intrinsic",
        "irl.intrinsic.icm",
        "irl.intrinsic.rnd",
        "irl.intrinsic.ride",
        "irl.intrinsic.riac",
        "irl.intrinsic.proposed",
        "irl.algo",
        "irl.algo.ppo",
        "irl.algo.advantage",
        "irl.data",
        "irl.data.storage",
        "irl.train",
        "irl.eval",
        "irl.sweep",
        "irl.plot",
    ]:
        assert importlib.import_module(mod) is not None
