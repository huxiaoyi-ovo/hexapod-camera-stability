"""Path checks shared by the offline analysis commands."""
from contextlib import contextmanager
from pathlib import Path
from stat import S_IMODE
from tempfile import NamedTemporaryFile
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


@contextmanager
def atomic_output_path(output: Path):
    """Publish a regenerated result only after its entire file is written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, prefix=f'.{output.name}.',
                            suffix='.tmp', delete=False) as file:
        temporary = Path(file.name)
    try:
        yield temporary
        try:
            mode = output.stat().st_mode
        except FileNotFoundError:
            pass
        else:
            temporary.chmod(S_IMODE(mode))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
