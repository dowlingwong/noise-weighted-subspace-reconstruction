from pathlib import Path

from src.noise_geometry.utils.paths import DEFAULT_DATA_ROOT, resolve_data_root


def test_data_root_precedence(monkeypatch, tmp_path):
    config_root = tmp_path / "config"
    env_root = tmp_path / "env"
    cli_root = tmp_path / "cli"
    monkeypatch.setenv("PAPER1_DATA_ROOT", str(env_root))

    assert resolve_data_root(cli_root, {"data_root": str(config_root)}) == cli_root.resolve()
    assert resolve_data_root(None, {"data_root": str(config_root)}) == env_root.resolve()
    monkeypatch.delenv("PAPER1_DATA_ROOT")
    assert resolve_data_root(None, {"data_root": str(config_root)}) == config_root.resolve()
    assert resolve_data_root() == Path(DEFAULT_DATA_ROOT)
