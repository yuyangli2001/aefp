from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from utils.checkpoint_paths import direct_ckpt_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Resolve the expected model filenames relative to the project.
def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def require_checkpoint(manifest: Mapping[str, Any]) -> str:
    path = resolve_path(manifest['path'])
    if not path.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    for config in manifest.get('config_files', ()):
        config_path = resolve_path(config['path'])
        if not config_path.is_file():
            raise FileNotFoundError(f'Missing model configuration: {config_path}')
    return str(path)


def fixed_ae_paths(manifest: Mapping[str, Any], *, solver: str = 'jinit') -> list[Path]:
    directory = resolve_path(manifest['path'])
    paths = [directory / item for item in manifest.get('stage_paths', ())]
    if solver == 'jinit':
        if len(paths) != 3:
            raise ValueError('JInit requires three stage_paths in recovery order.')
        return paths
    if solver in {'direct', 'feedforward'}:
        return paths[-1:] or [direct_ckpt_path(directory)]
    raise ValueError(f'Unsupported checkpoint solver: {solver}')
