import pytest

from ragkb.loaders import load_documents, load_file


def test_loads_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("你好世界", encoding="utf-8")
    (tmp_path / "b.md").write_text("# 标题\n内容", encoding="utf-8")
    result = load_documents(tmp_path)
    assert len(result.documents) == 2
    assert {d.metadata["source"] for d in result.documents} == {"a.txt", "b.md"}
    assert all(d.metadata["page"] is None for d in result.documents)
    assert result.warnings == []


def test_subdirectory_source_is_relative(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("内容", encoding="utf-8")
    result = load_documents(tmp_path)
    assert result.documents[0].metadata["source"] == "sub/c.txt"


def test_unknown_extension_skipped(tmp_path):
    (tmp_path / "a.xyz").write_text("x", encoding="utf-8")
    result = load_documents(tmp_path)
    assert result.documents == []
    assert any("a.xyz" in w for w in result.warnings)


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_load_file_unsupported(tmp_path):
    p = tmp_path / "a.xyz"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持"):
        load_file(p)
