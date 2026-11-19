"""Project-relative paths and output directory creation."""
from pathlib import Path

# Resolve paths independently of the caller's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def from_root(*relative_path_parts) -> Path:
    """Join path components relative to the project root."""
    return PROJECT_ROOT.joinpath(*relative_path_parts)


# Asset references use their current filenames, without historical path substitution.
def local_path(value) -> Path:
    path = from_root(str(value).replace('\\', '/')).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f'Use an asset path inside this project: {value}')
    return path


def ensure_dir(path: Path):
    """Create a directory and its parents when absent."""
    Path(path).mkdir(parents=True, exist_ok=True)
