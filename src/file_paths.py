"""Path checks shared by the offline analysis commands."""
from pathlib import Path
from typing import Iterable


def validate_output_path(output: Path, inputs: Iterable[Path]) -> Path:
    """Resolve an output path and reject aliases of any input file."""
    output = output.resolve()
    output_exists = output.exists()
    for path in inputs:
        source = path.resolve()
        if output == source or (output_exists and source.exists() and output.samefile(source)):
            raise ValueError(f'Output would overwrite input: {source}')
    return output
