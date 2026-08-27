from __future__ import annotations

import warnings

import numpy as np
import pytest

from compresso import TopKSAEConfig, TopKSAETrainer


class RecordingLogger:
    """Minimal duck-typed sink: the only contract is ``info(str)``."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, message: str) -> None:
        self.lines.append(message)


class RaisingLogger:
    def __init__(self) -> None:
        self.calls = 0

    def info(self, message: str) -> None:
        self.calls += 1
        raise RuntimeError("log sink is down")


def _config(**overrides) -> TopKSAEConfig:
    base = dict(hidden_dim=8, k=2, epochs=3, batch_size=4, show_progress=False, seed=0)
    base.update(overrides)
    return TopKSAEConfig(**base)


def _embeddings(rows: int = 12, dim: int = 5) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(rows, dim)).astype(np.float32)


def _epoch_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("[TopKSAE] epoch") and "step" not in line]


def test_no_logger_leaves_history_and_output_untouched():
    """The default path must not notice that logging exists."""
    embeddings = _embeddings()
    quiet = TopKSAETrainer(_config()).fit(embeddings)
    assert len(quiet.history) == 3
    assert quiet.logger is None


def test_progress_bar_is_unchanged_without_a_logger():
    """tqdm is a soft import, so only its absence is guaranteed here."""
    pytest.importorskip("tqdm")
    trainer = TopKSAETrainer(_config(show_progress=True))
    iterable = range(3)
    assert trainer._progress(iterable, total=3) is not iterable


def test_show_progress_false_stays_a_bare_iterable():
    trainer = TopKSAETrainer(_config(show_progress=False))
    iterable = range(3)
    assert trainer._progress(iterable, total=3) is iterable


@pytest.mark.parametrize("show_progress", [True, False])
def test_a_logger_suppresses_the_progress_bar(show_progress):
    """The two would report the same numbers, so the logger wins."""
    trainer = TopKSAETrainer(_config(show_progress=show_progress), logger=RecordingLogger())
    iterable = range(3)
    assert trainer._progress(iterable, total=3) is iterable


def test_three_epoch_fit_emits_start_epochs_and_end():
    logger = RecordingLogger()
    TopKSAETrainer(_config(validation_frac=0.25), logger=logger).fit(_embeddings())

    starts = [line for line in logger.lines if "fit started" in line]
    ends = [line for line in logger.lines if "fit finished" in line]
    epochs = _epoch_lines(logger.lines)
    assert len(starts) == 1
    assert len(epochs) == 3
    assert len(ends) == 1
    assert all(line.startswith("[TopKSAE] ") for line in logger.lines)


def test_epoch_lines_carry_the_tuning_diagnostics():
    logger = RecordingLogger()
    TopKSAETrainer(_config(validation_frac=0.25), logger=logger).fit(_embeddings())

    for line in _epoch_lines(logger.lines):
        assert "dead_features:" in line
        assert "active_count:" in line
        assert "val_loss:" in line


def test_start_line_describes_the_run():
    logger = RecordingLogger()
    TopKSAETrainer(
        _config(validation_frac=0.25, patience=2, standard_scaler_mean=True, noise_type="gaussian"),
        logger=logger,
    ).fit(_embeddings(rows=12, dim=5))

    start = next(line for line in logger.lines if "fit started" in line)
    for fragment in [
        "input_dim 5",
        "hidden_dim 8",
        "k 2",
        "train rows",
        "validation rows",
        "batches of 4",
        "epochs 3",
        "patience 2",
        "device cpu",
        "corruption gaussian",
        "standard scaling mean=True",
    ]:
        assert fragment in start, fragment


def test_a_new_history_key_appears_without_touching_the_formatter():
    """The epoch line dumps the record, so future metrics come along for free."""
    logger = RecordingLogger()
    trainer = TopKSAETrainer(_config(epochs=1), logger=logger)
    trainer._log_epoch(
        1, 1, {"epoch": 1.0, "loss": 0.5, "a_metric_added_later": 1.25}, 0.0, 0.0
    )

    line = logger.lines[-1]
    assert "a_metric_added_later: 1.2500" in line
    # The epoch number heads the line rather than repeating as a metric.
    assert "epoch:" not in line


def test_small_values_keep_their_magnitude():
    logger = RecordingLogger()
    trainer = TopKSAETrainer(_config(), logger=logger)
    trainer._log_epoch(1, 1, {"loss": 1e-7, "cosine_loss": 0.25}, 0.0, 0.0)

    line = logger.lines[-1]
    assert "loss: 1.0000e-07" in line
    assert "cosine_loss: 0.2500" in line


def test_end_line_reports_early_stopping():
    logger = RecordingLogger()
    trainer = TopKSAETrainer(_config(epochs=20, validation_frac=0.25, patience=1), logger=logger)
    trainer.fit(_embeddings())

    end = next(line for line in logger.lines if "fit finished" in line)
    assert "epochs_run" in end
    assert "best_epoch" in end
    assert "best_val_loss" in end
    if trainer.stopped_epoch is not None:
        assert f"early stopping fired at epoch {trainer.stopped_epoch}" in end
    else:
        assert "early stopping did not fire" in end


def test_end_line_says_so_when_early_stopping_did_not_fire():
    logger = RecordingLogger()
    TopKSAETrainer(_config(), logger=logger).fit(_embeddings())
    end = next(line for line in logger.lines if "fit finished" in line)
    assert "early stopping did not fire" in end


def test_step_logging_emits_one_line_per_batch():
    logger = RecordingLogger()
    # 12 rows at batch_size 4 is 3 batches per epoch, over 3 epochs.
    TopKSAETrainer(_config(log_every_n_steps=1), logger=logger).fit(_embeddings(rows=12))

    steps = [line for line in logger.lines if " step " in line]
    assert len(steps) == 9
    assert "step 1/3" in steps[0]
    assert "step 3/3" in steps[2]


def test_step_logging_honours_the_interval():
    logger = RecordingLogger()
    TopKSAETrainer(_config(log_every_n_steps=2), logger=logger).fit(_embeddings(rows=12))

    steps = [line for line in logger.lines if " step " in line]
    # Only batch 2 of each epoch's three hits the interval.
    assert len(steps) == 3
    assert all("step 2/3" in line for line in steps)


def test_step_logging_is_off_by_default():
    logger = RecordingLogger()
    TopKSAETrainer(_config(), logger=logger).fit(_embeddings())
    assert not [line for line in logger.lines if " step " in line]


def test_negative_step_interval_is_rejected():
    with pytest.raises(ValueError, match="log_every_n_steps must be >= 0"):
        TopKSAETrainer(_config(log_every_n_steps=-1)).build(input_dim=5)


def test_a_raising_logger_does_not_end_the_fit():
    """Hours of training must not be lost to a broken log handler."""
    logger = RaisingLogger()
    with pytest.warns(RuntimeWarning, match="logging disabled"):
        trainer = TopKSAETrainer(_config(log_every_n_steps=1), logger=logger).fit(_embeddings())

    assert len(trainer.history) == 3
    # Disabled after the first failure rather than raising once per line.
    assert logger.calls == 1


def test_log_prefix_is_configurable():
    logger = RecordingLogger()
    TopKSAETrainer(_config(log_prefix="SAE"), logger=logger).fit(_embeddings())
    assert all(line.startswith("[SAE] ") for line in logger.lines)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "0s"), (2.5, "2s"), (0.25, "250ms"), (1e-5, "10us")],
)
def test_durations_are_scaled_to_readable_units(seconds, expected):
    assert TopKSAETrainer._format_duration(seconds) == expected


def test_durations_can_carry_a_unit_name():
    assert TopKSAETrainer._format_duration(2.0, "epoch") == "2s/epoch"


def test_non_numeric_metrics_still_print():
    logger = RecordingLogger()
    trainer = TopKSAETrainer(_config(), logger=logger)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trainer._log_epoch(1, 1, {"note": "restored"}, 0.0, 0.0)
    assert "note: restored" in logger.lines[-1]


def test_a_strict_warning_filter_still_does_not_end_the_fit():
    """``-W error`` must not turn the courtesy notice into a lost fit."""
    logger = RaisingLogger()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trainer = TopKSAETrainer(_config(), logger=logger).fit(_embeddings())

    assert len(trainer.history) == 3
    assert logger.calls == 1


@pytest.mark.parametrize("method", ["encode", "reconstruct", "transform"])
def test_inference_passes_are_logged(method):
    """Suppressing tqdm must not leave these paths reporting nothing at all."""
    logger = RecordingLogger()
    trainer = TopKSAETrainer(_config(), logger=logger).fit(_embeddings())
    logger.lines.clear()

    getattr(trainer, method)(_embeddings())

    assert any(line.startswith(f"[TopKSAE] {method} started:") for line in logger.lines)
    assert any(line.startswith(f"[TopKSAE] {method} finished:") for line in logger.lines)


def test_inference_step_lines_honour_the_interval():
    logger = RecordingLogger()
    trainer = TopKSAETrainer(_config(log_every_n_steps=1), logger=logger).fit(_embeddings())
    logger.lines.clear()

    trainer.transform(_embeddings(rows=12))

    steps = [line for line in logger.lines if " step " in line]
    assert len(steps) == 3
    assert "transform step 1/3" in steps[0]


def test_fit_transform_reports_its_transform_phase():
    """The fit's end line must not be the last thing a long run says."""
    logger = RecordingLogger()
    TopKSAETrainer(_config(), logger=logger).fit_transform(_embeddings())

    fit_end = next(i for i, line in enumerate(logger.lines) if "fit finished" in line)
    assert any("transform started" in line for line in logger.lines[fit_end:])
    assert "transform finished" in logger.lines[-1]


@pytest.mark.parametrize("method", ["encode", "reconstruct", "transform"])
def test_inference_keeps_its_progress_bar_without_a_logger(method):
    """tqdm still wraps these passes when no logger is attached."""
    pytest.importorskip("tqdm")
    trainer = TopKSAETrainer(_config(show_progress=True)).fit(_embeddings())
    calls: list[object] = []
    original = trainer._progress

    def spy(iterable, **kwargs):
        calls.append(iterable)
        return original(iterable, **kwargs)

    trainer._progress = spy
    getattr(trainer, method)(_embeddings())
    assert len(calls) == 1


@pytest.mark.parametrize("method", ["encode", "reconstruct", "transform"])
def test_inference_results_are_unchanged_by_logging(method):
    """The logged pass is a wrapper, so it must not touch what comes out."""
    embeddings = _embeddings()
    quiet = TopKSAETrainer(_config(), logger=None).fit(embeddings)
    loud = TopKSAETrainer(_config(), logger=RecordingLogger()).fit(embeddings)
    loud.sae.load_state_dict(quiet.sae.state_dict())

    left = getattr(quiet, method)(embeddings)
    right = getattr(loud, method)(embeddings)
    if hasattr(left, "to_dense"):
        left, right = left.to_dense(), right.to_dense()
    assert np.allclose(left.numpy(), right.numpy())
