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
    last_checkpint_path = sorted(
        Path(path_to_checkpoints).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )[-1]
    with open(last_checkpint_path / "trainer_state.json", "r") as file_handler:
        train_state = json.load(file_handler)
    best_model_checkpint_path = train_state["best_model_checkpoint"]

    return best_model_checkpint_path
