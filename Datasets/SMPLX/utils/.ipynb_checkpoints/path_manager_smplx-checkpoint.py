from pathlib import Path

class PathManager_SMPLX():
    def __init__(self) -> None:
        cur_path = Path(__file__)

        self.root_dataset = cur_path.parent.parent
        print(self.root_dataset)
        assert (self.root_dataset.exists()), "Something unexpected happened. Root path does not exist."

