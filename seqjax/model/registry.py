
@dataclass
class DataConfig:
    parameter_setting: str
    sequence_length: int
    seed: int
    target: *ValidTarget*

    @property
    def dataset_name(self):
        dataset_name = sequential_models[self.target].label
        dataset_name += f"-{self.parameter_setting}"
        dataset_name += f"-d{self.seed}"
        dataset_name += f"-l{self.sequence_length}"
        return dataset_name

