import json
from pathlib import Path


def test_raskolnikov_rag_assets_exist() -> None:
    root = Path("Big_DATA/Raskolnikov/RAG")
    assert (root / "metadata" / "raskolnikov_character_profile.md").exists()
    assert (root / "metadata" / "raskolnikov_timeline.json").exists()
    assert (root / "metadata" / "raskolnikov_persona_guide.md").exists()
    assert (root / "metadata" / "raskolnikov_key_themes.md").exists()
    assert (root / "metadata" / "raskolnikov_relationships.md").exists()
    assert (root / "metadata" / "raskolnikov_tr_en_glossary.md").exists()
    assert (root / "metadata" / "raskolnikov_tr_en_title_aliases.md").exists()
    assert (root / "metadata" / "raskolnikov_tr_en_retrieval_bridges.md").exists()
    assert (root / "sources" / "source_manifest.json").exists()


def test_raskolnikov_timeline_json_is_valid() -> None:
    path = Path("Big_DATA/Raskolnikov/RAG/metadata/raskolnikov_timeline.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) > 0
