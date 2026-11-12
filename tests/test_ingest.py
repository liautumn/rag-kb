from ragkb.config import Settings
from ragkb.ingest import main


def test_missing_docs_dir_returns_1(monkeypatch, tmp_path, capsys):
    settings = Settings(docs_dir=tmp_path / "nope")
    monkeypatch.setattr("ragkb.ingest.get_settings", lambda: settings)
    assert main([]) == 1
    assert "文档目录不存在" in capsys.readouterr().out


def test_empty_docs_dir_returns_1(monkeypatch, tmp_path, capsys):
    settings = Settings(docs_dir=tmp_path)
    monkeypatch.setattr("ragkb.ingest.get_settings", lambda: settings)
    assert main([]) == 1
    assert "未找到可加载的文档" in capsys.readouterr().out
