import os
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer


STATE_FILE = "early_stopping.json"


def restore_early_stopping(trainer: "BaseTrainer") -> None:
    from ultralytics.utils import LOGGER, RANK

    if RANK not in {-1, 0} or not trainer.resume:
        return

    state_path = Path(trainer.save_dir) / STATE_FILE
    epoch_key = str(trainer.start_epoch)
    if not state_path.exists():
        LOGGER.warning(
            "No saved early-stopping history found; this resume will start "
            "with a fresh patience counter."
        )
        return

    history = json.loads(state_path.read_text(encoding="utf-8"))
    if epoch_key not in history:
        raise ValueError(
            f"{state_path} has no early-stopping state for epoch {epoch_key}. "
            "Resume from a checkpoint with matching saved history."
        )

    state = history[epoch_key]
    best_epoch = state["best_epoch"]
    best_fitness = state["best_fitness"]
    if (
        type(best_epoch) is not int
        or not 0 <= best_epoch <= trainer.start_epoch
        or type(best_fitness) not in {int, float}
    ):
        raise ValueError(f"Invalid early-stopping state in {state_path}")

    trainer.stopper.best_epoch = best_epoch
    trainer.stopper.best_fitness = best_fitness
    elapsed = trainer.start_epoch - best_epoch
    trainer.stopper.possible_stop = elapsed >= trainer.stopper.patience - 1
    LOGGER.info(f"Restored early stopping: {elapsed} epochs without improvement.")


def save_early_stopping(trainer: "BaseTrainer") -> None:
    from ultralytics.utils import RANK

    if RANK not in {-1, 0}:
        return

    state_path = Path(trainer.save_dir) / STATE_FILE
    completed_epoch = trainer.epoch + 1
    history = {}
    if state_path.exists():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        history = {
            key: value for key, value in saved.items() if int(key) < completed_epoch
        }

    history[str(completed_epoch)] = {
        "best_epoch": trainer.stopper.best_epoch,
        "best_fitness": float(trainer.stopper.best_fitness),
        "epochs_without_improvement": completed_epoch - trainer.stopper.best_epoch,
    }
    temporary_path = state_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(history, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary_path.replace(state_path)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    config_dir = project_dir / ".yolo-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

    from ultralytics import settings
    from ultralytics.utils.callbacks.base import default_callbacks
    settings.update({"tensorboard": True})
    default_callbacks["on_train_start"].append(restore_early_stopping)
    default_callbacks["on_fit_epoch_end"].append(save_early_stopping)

    from ultralytics.cfg import entrypoint

    entrypoint()


if __name__ == "__main__":
    main()
