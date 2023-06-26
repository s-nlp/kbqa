from pathlib import Path
import json


def get_best_checkpoint_path(path_to_checkpoints: str) -> str:
    """get_best_checkpoint_path - helper for loading path to best HF trained model
    helper will extract path to best checkpoint from directory with HF checkpoints

    Args:
        path_to_checkpoints (str): path to directory with checkpoints of trained model

    Returns:
        str: path to best checkpoint directory
    """
    if (Path(path_to_checkpoints) / "checkpoint-best").exists():
        return Path(path_to_checkpoints) / "checkpoint-best"
    else:
        pathes = sorted(
            Path(path_to_checkpoints).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if len(pathes) == 0:
            return None

        last_checkpint_path = pathes[-1]
        with open(last_checkpint_path / "trainer_state.json", "r") as file_handler:
            train_state = json.load(file_handler)
        best_model_checkpoint = Path(train_state["best_model_checkpoint"]).name

        return path_to_checkpoints / best_model_checkpoint
