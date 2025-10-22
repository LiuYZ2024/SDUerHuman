from pathlib import Path

class PathManager_SMPLX():
    def __init__(self) -> None:
        cur_path = Path(__file__)
        # print(cur_path)
        self.root_dataset = cur_path.parent.parent
        # print(self.root_dataset)
        assert (self.root_dataset.exists()), "Something unexpected happened. Root path does not exist."

        self.outputs = self.root_dataset / 'data_output'
        assert (self.outputs.exists()), "Something unexpected happened. Outputs path does not exist."

# pm = PathManager_SMPLX()
# print(pm.outputs)