import json
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path


class CLIParams:
    def __post_init__(self) -> None:
        """Override params using cli args, create output dir, and save params."""
        self.cli_override()
        self.check_args()
        self.create_output_dir_and_save()

    def cli_override(self) -> None:
        """Override params from CLI args."""
        parser = ArgumentParser()
        for k, v in asdict(self).items():
            parser.add_argument(f"--{k}", type=type(v), default=v)
        args = parser.parse_args()

        for k, v in vars(args).items():
            setattr(self, k, v)

    def create_output_dir_and_save(self) -> None:
        """Create the output dir and save params."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "hyper_params.json", mode="w") as f:
            json.dump(asdict(self), f, indent=4)

        self.output_dir = output_dir  # HACK: set after json dump b/c Path is not serializable.

    def check_args(self):
        """Check args are valid."""
        pass