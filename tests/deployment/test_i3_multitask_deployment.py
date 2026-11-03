"""Unit tests for multitask / multi-model deployment in icetray."""

import os
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from graphnet.constants import TEST_DATA_DIR
from graphnet.models import StandardModel
from graphnet.models.data_representation import EdgelessGraph
from graphnet.models.data_representation.graphs.nodes import (
    ClusterSummaryFeatures,
)
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import NeutrinoEventMultitaskTransformer
from graphnet.models.task import IdentityTask
from graphnet.training.loss_functions import MSELoss

icecube = pytest.importorskip("icecube")

from icecube import dataio, icetray  # noqa: E402

from graphnet.data.extractors.icecube import (  # noqa: E402
    I3FeatureExtractorIceCube86,
)
from graphnet.deployment.icecube import (  # noqa: E402
    I3InferenceModule,
    I3MultipleModelInferenceModule,
    I3ParticleInferenceModule,
)

_NAMES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge"]
# five physics frames with SRTInIcePulses, and their GCD
I3_DIR = os.path.join(TEST_DATA_DIR, "i3", "oscNext_genie_level3_v02")
I3_FILE = os.path.join(I3_DIR, "oscNext_genie_level3_v02_first_5_frames.i3.gz")
GCD_FILE = os.path.join(
    I3_DIR,
    "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz",
)
MULTITASK_COLUMNS = ["PosX", "PosY", "PosZ", "Time", "DirX", "DirY", "DirZ"]


def _model(single_output: bool = False, reference: bool = True) -> Any:
    """Tiny multitask model; the time is relative to `reference_time`."""
    graph_definition = EdgelessGraph(
        detector=IceCube86(replace_with_identity=["dom_time", "charge"]),
        node_definition=ClusterSummaryFeatures(
            cluster_on=_NAMES[:3],
            input_feature_names=_NAMES,
            charge_after_t=[10],
            time_after_charge_pct=[10, 50],
            reference_charge_quantile=0.1,
            reference_time_attribute="reference_time",
        ),
        input_feature_names=_NAMES,
    )
    backbone = NeutrinoEventMultitaskTransformer(
        n_attention_blocks=1,
        n_rel=0,
        hidden_dim=16,
        num_heads=2,
        n_features=graph_definition.nb_outputs,
        out_dim=8,
    )
    if single_output:
        tasks = [
            IdentityTask(
                nb_outputs=1,
                hidden_size=8,
                target_labels=["score"],
                prediction_labels=["Score"],
                loss_function=MSELoss(),
            )
        ]
    else:
        tasks = [
            IdentityTask(
                nb_outputs=4,
                hidden_size=8,
                target_labels=["x", "y", "z", "t"],
                prediction_labels=MULTITASK_COLUMNS[:4],
                reference_labels=(
                    {"Time": "reference_time"} if reference else None
                ),
                loss_function=MSELoss(),
                # a time of O(100 ns) relative to the reference
                transform_inference=lambda x: x * 100,
                transform_target=lambda x: x / 100,
            ),
            IdentityTask(
                nb_outputs=3,
                hidden_size=8,
                target_labels=["dx", "dy", "dz"],
                prediction_labels=MULTITASK_COLUMNS[4:],
                loss_function=MSELoss(),
            ),
        ]
    return StandardModel(
        data_representation=graph_definition, backbone=backbone, tasks=tasks
    )


def _save(model: Any, tmp_path: Any, name: str) -> Dict[str, Any]:
    config = str(tmp_path / f"{name}.yml")
    weights = str(tmp_path / f"{name}.pth")
    model.save_config(config)
    model.save_state_dict(weights)
    return {"model_config": config, "state_dict": weights}


def _run(module: Any, prefixes: tuple) -> List[Dict[str, Any]]:
    """Run `module` on the test file; return the physics frames' objects whose
    keys start with one of `prefixes`."""
    from icecube.icetray import I3Tray

    frames: List[Dict[str, Any]] = []
    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", FilenameList=[GCD_FILE, I3_FILE])
    tray.AddModule(module)
    tray.Add(
        lambda frame: frames.append(
            {k: frame[k] for k in frame.keys() if k.startswith(prefixes)}
        ),
        Streams=[icetray.I3Frame.Physics],
    )
    tray.Execute()
    tray.Finish()
    assert len(frames) == 5
    return frames


def _extractor() -> Any:
    return I3FeatureExtractorIceCube86("SRTInIcePulses")


def test_all_tasks_of_several_models_with_absolute_time(tmp_path: Any) -> None:
    """Every task is written; the time is absolute whether the model adds the
    reference (reference_labels) or the module does (reference_times)."""
    torch.manual_seed(0)
    model = _model()
    absolute = _save(model, tmp_path, "absolute")
    relative_model = _model(reference=False)
    relative_model.load_state_dict(model.state_dict())
    relative = _save(relative_model, tmp_path, "relative")

    frames = _run(
        I3InferenceModule(
            pulsemap_extractor=_extractor(),
            model_config=[absolute["model_config"], relative["model_config"]],
            state_dict=[absolute["state_dict"], relative["state_dict"]],
            model_name=["nemt", "nemt_rel"],
            gcd_file=GCD_FILE,
            reference_times=[None, {"Time": "reference_time"}],
            record_runtimes=True,
        ),
        ("nemt",),
    )
    for frame in frames:
        for name in ["nemt", "nemt_rel"]:
            for column in MULTITASK_COLUMNS + ["reference_time"]:
                assert f"{name}_{column}" in frame, f"{name}_{column}"
        reference = frame["nemt_reference_time"].value
        assert reference == frame["nemt_rel_reference_time"].value
        # absolute: ~ the event time, not O(100 ns)
        assert abs(frame["nemt_Time"].value - reference) < 1e3 < reference
        assert np.isclose(
            frame["nemt_Time"].value, frame["nemt_rel_Time"].value, atol=1e-2
        )
        assert set(frame["nemt_Speed"].keys()) == {
            "data_repr_time",
            "inference_time",
            "write_time",
            "total_time",
        }

    with pytest.raises(ValueError, match="twice"):
        I3InferenceModule(
            pulsemap_extractor=_extractor(),
            gcd_file=GCD_FILE,
            model_name="nemt",
            reference_times={"Time": "reference_time"},
            **absolute,
        )


def test_particle_module(tmp_path: Any) -> None:
    """`<name>_particle` holds the configured columns, which are then not
    written as I3Doubles."""
    torch.manual_seed(1)
    saved = _save(_model(), tmp_path, "nemt")
    kwargs = dict(pulsemap_extractor=_extractor(), gcd_file=GCD_FILE, **saved)
    plain = _run(I3InferenceModule(model_name="plain", **kwargs), ("plain",))
    particles = _run(
        I3ParticleInferenceModule(
            direction=["DirX", "DirY", "DirZ"],
            position=["PosX", "PosY", "PosZ"],
            time="Time",
            energy="DirX",  # any column: the test model has no energy
            model_name="nemt",
            **kwargs,
        ),
        ("nemt",),
    )
    for p_frame, frame in zip(plain, particles):
        particle = frame["nemt_particle"]
        assert particle.fit_status == particle.FitStatus.OK
        assert particle.shape == particle.ParticleShape.InfiniteTrack
        assert np.isclose(particle.time, p_frame["plain_Time"].value)
        assert np.isclose(particle.pos.z, p_frame["plain_PosZ"].value)
        assert np.isclose(particle.energy, p_frame["plain_DirX"].value)
        direction = np.array(
            [p_frame[f"plain_Dir{c}"].value for c in "XYZ"], dtype=float
        )
        direction /= np.linalg.norm(direction)
        assert np.isclose(particle.dir.z, direction[2], atol=1e-6)
        assert not any(f"nemt_{c}" in frame for c in MULTITASK_COLUMNS)
        assert "nemt_reference_time" in frame


def test_multiple_model_map(tmp_path: Any) -> None:
    """One I3MapStringDouble: bare name for a single-output model, one float
    per column (and reference) otherwise."""
    torch.manual_seed(2)
    single = _save(_model(single_output=True), tmp_path, "score")
    multi = _save(_model(), tmp_path, "nemt")
    kwargs = dict(
        key_name="Scores",
        pulsemap_extractor=_extractor(),
        gcd_file=GCD_FILE,
        model_config=[single["model_config"], multi["model_config"]],
        state_dict=[single["state_dict"], multi["state_dict"]],
        model_name=["bundle", "nemt"],
    )
    frames = _run(I3MultipleModelInferenceModule(**kwargs), ("Scores",))
    for frame in frames:
        scores = dict(frame["Scores"])
        expected = {"bundle", "nemt_reference_time"} | {
            f"nemt_{c}" for c in MULTITASK_COLUMNS
        }
        assert set(scores) == expected
        assert all(isinstance(v, float) for v in scores.values())

    # first_column_only: `<model_name>` -> the first column only (the map
    # layout deployed bundle-classifier selections read)
    first = _run(
        I3MultipleModelInferenceModule(first_column_only=True, **kwargs),
        ("Scores",),
    )
    for frame, all_columns in zip(first, frames):
        scores = dict(frame["Scores"])
        assert set(scores) == {"bundle", "nemt"}
        assert scores["bundle"] == all_columns["Scores"]["bundle"]
        assert scores["nemt"] == all_columns["Scores"]["nemt_PosX"]
