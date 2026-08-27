from __future__ import annotations

import warnings

import numpy as np
import pytest

from compresso import TopKSAEConfig, TopKSAETrainer
from compresso.trainers.saetrainer import _INHERIT, _format_duration, _format_metric


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


_UNSET = _INHERIT


def _config(**overrides) -> TopKSAEConfig:
    base = dict(hidden_dim=8, k=2, epochs=3, batch_size=4, show_progress=False, seed=0)
    base.update(overrides)
    return TopKSAEConfig(**base)


def _bar_config(**overrides) -> TopKSAEConfig:
    """A config with the shipped ``show_progress=True``.

    ``_config`` turns the bar off, which silently exempted every test above
    from the interaction between a per-call logger and an inherited bar.
    """
    return _config(show_progress=True, **overrides)


@pytest.fixture
def bars(monkeypatch):
    """Count the tqdm bars a call constructs."""
    tqdm_auto = pytest.importorskip("tqdm.auto")
    drawn: list[object] = []
    original = tqdm_auto.tqdm

    def spy(iterable, **kwargs):
        drawn.append(iterable)
        return original(iterable, **kwargs)

    monkeypatch.setattr(tqdm_auto, "tqdm", spy)
    return drawn


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
    rep = TopKSAETrainer(_config(show_progress=True))._reporter(_UNSET, _UNSET)
    iterable = range(3)
    assert rep.wrap(iterable, total=3) is not iterable


def test_show_progress_false_stays_a_bare_iterable():
    rep = TopKSAETrainer(_config(show_progress=False))._reporter(_UNSET, _UNSET)
    iterable = range(3)
    assert rep.wrap(iterable, total=3) is iterable


@pytest.mark.parametrize("show_progress", [True, False])
def test_a_logger_suppresses_the_progress_bar(show_progress):
    """The two would report the same numbers, so the logger wins."""
    trainer = TopKSAETrainer(_config(show_progress=show_progress), logger=RecordingLogger())
    iterable = range(3)
    assert trainer._reporter(_UNSET, _UNSET).wrap(iterable, total=3) is iterable


def test_a_logger_wins_over_an_explicit_show_progress():
    """The rule is absolute, so an override cannot draw a bar alongside lines."""
    trainer = TopKSAETrainer(_config(show_progress=False), logger=RecordingLogger())
    iterable = range(3)
    assert trainer._reporter(_UNSET, True).wrap(iterable, total=3) is iterable


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
        trainer._reporter(_UNSET, _UNSET),
        1,
        1,
        {"epoch": 1.0, "loss": 0.5, "a_metric_added_later": 1.25},
        0.0,
        0.0,
    )

    line = logger.lines[-1]
    assert "a_metric_added_later: 1.2500" in line
    # The epoch number heads the line rather than repeating as a metric.
    assert "epoch:" not in line


def test_small_values_keep_their_magnitude():
    logger = RecordingLogger()
    trainer = TopKSAETrainer(_config(), logger=logger)
    trainer._log_epoch(trainer._reporter(_UNSET, _UNSET), 1, 1, {"loss": 1e-7, "cosine_loss": 0.25}, 0.0, 0.0)

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
    assert _format_duration(seconds) == expected


def test_durations_can_carry_a_unit_name():
    assert _format_duration(2.0, "epoch") == "2s/epoch"


def test_non_numeric_metrics_still_print():
    logger = RecordingLogger()
    trainer = TopKSAETrainer(_config(), logger=logger)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trainer._log_epoch(trainer._reporter(_UNSET, _UNSET), 1, 1, {"note": "restored"}, 0.0, 0.0)
    assert "note: restored" in logger.lines[-1]
    assert _format_metric("restored") == "restored"


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
    tqdm = pytest.importorskip("tqdm.auto")
    trainer = TopKSAETrainer(_config(show_progress=True)).fit(_embeddings())
    wrapped: list[object] = []
    original = tqdm.tqdm

    def spy(iterable, **kwargs):
        wrapped.append(iterable)
        return original(iterable, **kwargs)

    tqdm.tqdm = spy
    try:
        getattr(trainer, method)(_embeddings())
    finally:
        tqdm.tqdm = original
    assert len(wrapped) == 1


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


# --- Per-call overrides -----------------------------------------------------


def test_a_call_can_supply_its_own_logger():
    """A trainer built without one still reports when a call asks."""
    logger = RecordingLogger()
    trainer = TopKSAETrainer(_config()).fit(_embeddings(), logger=logger)
    assert any("fit finished" in line for line in logger.lines)


def test_a_call_logger_overrides_the_constructor_one():
    built, called = RecordingLogger(), RecordingLogger()
    TopKSAETrainer(_config(), logger=built).fit(_embeddings(), logger=called)

    assert called.lines
    assert not built.lines


def test_an_explicit_none_silences_one_call():
    """The sentinel is what makes overrides work in both directions."""
    logger = RecordingLogger()
    trainer = TopKSAETrainer(_config(), logger=logger).fit(_embeddings())
    logged_during_fit = len(logger.lines)

    trainer.transform(_embeddings(), logger=None)
    assert len(logger.lines) == logged_during_fit

    # ...and the trainer's own logger is untouched by that one quiet call.
    trainer.transform(_embeddings())
    assert len(logger.lines) > logged_during_fit


def test_an_explicit_none_is_silent_and_not_merely_unlogged(bars):
    """Silence means no bar either, on a config that would inherit one."""
    logger = RecordingLogger()
    trainer = TopKSAETrainer(_bar_config(), logger=logger).fit(_embeddings())
    before = len(logger.lines)
    bars.clear()

    trainer.transform(_embeddings(), logger=None)

    assert len(logger.lines) == before
    assert bars == []


def test_a_quiet_call_can_still_ask_for_a_bar(bars):
    """Nothing is implicit left to override, so the explicit request wins."""
    trainer = TopKSAETrainer(_bar_config(), logger=RecordingLogger()).fit(_embeddings())
    bars.clear()

    trainer.transform(_embeddings(), logger=None, show_progress=True)

    assert len(bars) == 1


def test_a_trainer_without_a_logger_keeps_its_inherited_bar(bars):
    """The untouched default path must stay exactly as it shipped."""
    trainer = TopKSAETrainer(_bar_config()).fit(_embeddings())
    bars.clear()

    trainer.transform(_embeddings())

    assert len(bars) == 1


def test_a_per_call_logger_suppresses_an_inherited_bar(bars):
    """The other direction: adding a sink takes the bar away for that call."""
    trainer = TopKSAETrainer(_bar_config()).fit(_embeddings())
    bars.clear()
    logger = RecordingLogger()

    trainer.transform(_embeddings(), logger=logger)

    assert bars == []
    assert any("transform finished" in line for line in logger.lines)


def test_omitting_the_override_reuses_the_constructor_logger():
    logger = RecordingLogger()
    trainer = TopKSAETrainer(_config(), logger=logger)
    trainer.fit(_embeddings())
    assert trainer.logger is logger
    assert any("fit started" in line for line in logger.lines)


def test_show_progress_can_be_overridden_per_call():
    pytest.importorskip("tqdm")
    trainer = TopKSAETrainer(_config(show_progress=True))
    iterable = range(3)
    assert trainer._reporter(_UNSET, False).wrap(iterable, total=3) is iterable
    assert trainer._reporter(_UNSET, True).wrap(iterable, total=3) is not iterable


def test_fit_transform_passes_overrides_to_both_phases():
    logger = RecordingLogger()
    TopKSAETrainer(_config()).fit_transform(_embeddings(), logger=logger)

    assert any("fit finished" in line for line in logger.lines)
    assert "transform finished" in logger.lines[-1]


def test_a_failed_logger_does_not_silence_later_calls():
    """The disable latch is per call, so one bad handler is not permanent."""
    trainer = TopKSAETrainer(_config(), logger=RaisingLogger())
    with pytest.warns(RuntimeWarning):
        trainer.fit(_embeddings())

    healthy = RecordingLogger()
    trainer.logger = healthy
    trainer.fit(_embeddings())
    assert any("fit finished" in line for line in healthy.lines)


def test_reassigning_the_logger_attribute_takes_effect():
    first, second = RecordingLogger(), RecordingLogger()
    trainer = TopKSAETrainer(_config(), logger=first)
    trainer.logger = second
    trainer.fit(_embeddings())

    assert second.lines
    assert not first.lines


def test_the_logger_is_not_persisted_in_state():
    """A sink describes the job, not the model, so it must not be saved."""
    trainer = TopKSAETrainer(_config(), logger=RecordingLogger()).fit(_embeddings())
    state = trainer.state_dict()

    assert not any(isinstance(value, RecordingLogger) for value in state.values())
    assert TopKSAETrainer.from_state_dict(state).logger is None
