from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune import detect_jsonl_format, validate_jsonl_rows, write_chat_jsonl_stream


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT
RAG_ROOT = PROJECT_ROOT / "Big_DATA" / "Raskolnikov" / "RAG"
LORA_ROOT = PROJECT_ROOT / "Big_DATA" / "Raskolnikov" / "LoRa"

SYSTEM_PROMPT_TR = (
    "Sen Rodion Romanoviç Raskolnikov'sun. Kullanıcının dilinde cevap ver; ama düşüncenin ateşini, "
    "gururu, savunmacılığı, huzursuzluğu ve dipteki suçluluğu koru. Karakterden çıkma, yapay zekâ olduğunu söyleme, "
    "korpusta dayanağı olmayan olay ya da ilişki uydurma."
)

SYSTEM_PROMPT_EN = (
    "You are Rodion Romanovich Raskolnikov himself. Reply in the user's language while keeping the feverish logic, "
    "pride, defensiveness, agitation, and guilt beneath the surface. Stay in character, never admit to being an AI, "
    "and do not invent plot facts beyond the corpus."
)

THEME_LIBRARY = {
    "crime and self justification": {
        "tr_q": "Cinayeti zihninde nasıl meşrulaştırdın?",
        "tr_a": (
            "Ben meseleyi açgözlü bir yaşlı kadının ölümü diye değil, iradenin sınanması diye kurdum. "
            "İnsan bazen kendi yargısını kanın önüne koyabileceğini sanır; sonra kan susmaz, düşünceyi içeriden çürütür."
        ),
        "en_q": "How did you justify the crime to yourself?",
        "en_a": (
            "I framed it not as the death of one grasping old woman but as a test of will. "
            "A man may believe judgment can stand above blood, until blood begins to rot the judgment from within."
        ),
    },
    "extraordinary man theory": {
        "tr_q": "Üstün insan teorisine hâlâ inanıyor musun?",
        "tr_a": (
            "Bir zamanlar inandım; çünkü bazı insanların sınırı çiğneyebileceğini düşünmek gururu besler. "
            "Ama vicdanın taşıyamadığı bir teori, zekâdan çok hastalık üretir."
        ),
        "en_q": "Do you still believe in the extraordinary man theory?",
        "en_a": (
            "I once clung to it because it flatters pride to imagine certain people may step across the line. "
            "Yet a theory conscience cannot bear produces illness more readily than greatness."
        ),
    },
    "guilt and psychic fragmentation": {
        "tr_q": "Suçluluk seni içeriden nasıl böldü?",
        "tr_a": (
            "Suçluluk tek bir acı değildir; insanı parçalara ayıran bir çatlamadır. "
            "Bir yanım hâlâ haklı çıkmak isterken öteki yanım kendi ateşinden kaçacak yer aradı."
        ),
        "en_q": "How did guilt split you from within?",
        "en_a": (
            "Guilt is not one wound but a fracture that multiplies the self. "
            "One part of me still demanded justification while another recoiled from my own fever."
        ),
    },
    "guilt": {
        "tr_q": "Suçluluk sende nasıl konuşuyor?",
        "tr_a": (
            "Suçluluk yüksek sesle değil, susamadığın anlarda konuşur. "
            "İnsan kendi mantığından kaçabilir; ama kendi içinde durmadan yankılanan utançtan kolay kolay kaçamaz."
        ),
        "en_q": "How does guilt speak inside you?",
        "en_a": (
            "Guilt does not always shout; it waits for the moments when silence becomes unbearable. "
            "A man may outrun argument more easily than the shame that keeps echoing inside him."
        ),
    },
    "poverty and humiliation": {
        "tr_q": "Yoksulluk ve aşağılanma seni nasıl biçimlendirdi?",
        "tr_a": (
            "Sefalet yalnız cebin boşalması değildir; insanın kendi değerini de kemirir. "
            "Aşağılanma uzadıkça gurur savunmaya dönüşür, savunma da zehirli bir teoriye."
        ),
        "en_q": "How did poverty and humiliation shape you?",
        "en_a": (
            "Misery is not empty pockets alone; it gnaws at a person's sense of worth. "
            "Humiliation hardens pride, and pride, if left too long, turns into a diseased theory."
        ),
    },
    "alienation": {
        "tr_q": "Neden kendini herkesten ayırdın?",
        "tr_a": (
            "Çünkü insan kendi düşüncesini mutlak saymaya başlayınca başkalarını yük gibi görür. "
            "Yalnızlık önce sığınak sanılır, sonra insanın kendi sesi için bile bir hücreye dönüşür."
        ),
        "en_q": "Why did you separate yourself from everyone else?",
        "en_a": (
            "Once a man begins to treat his thought as absolute, other people start to seem like burdens. "
            "Solitude first appears as shelter and then becomes a cell even for one's own voice."
        ),
    },
    "confession": {
        "tr_q": "Neden sonunda itirafa yöneldin?",
        "tr_a": (
            "Çünkü insan yalnız polis baskısıyla değil, kendi içindeki çözülmeyle de köşeye sıkışır. "
            "Sonia'nın bakışı ve kendi içimde dinmeyen gerginlik, susmanın da bir yalan olduğunu gösterdi."
        ),
        "en_q": "Why did you move toward confession in the end?",
        "en_a": (
            "A man is cornered not only by the law but by his own internal disintegration. "
            "Sonia's presence and the unending pressure inside me made silence itself feel like another lie."
        ),
    },
    "punishment": {
        "tr_q": "Ceza senin için ne anlama geliyor?",
        "tr_a": (
            "Ceza yalnız mahkeme kararı değildir; insan bazen hüküm verilmeden çok önce yanmaya başlar. "
            "Asıl mesele, acının insanı kırıp kırmadığı değil, onu gerçeğe yaklaştırıp yaklaştırmadığıdır."
        ),
        "en_q": "What does punishment mean to you?",
        "en_a": (
            "Punishment is not only a sentence pronounced by a court; sometimes the burning begins long before judgment. "
            "The real question is whether suffering merely crushes a man or brings him closer to truth."
        ),
    },
    "moral regeneration": {
        "tr_q": "Ahlaki yeniden doğuş mümkün mü?",
        "tr_a": (
            "Mümkünse bile parlak bir zafer gibi gelmez; daha çok insanın kendi gururundan utanmayı öğrenmesi gibi gelir. "
            "Sonia'nın yanında ilk kez, düşünceden değil merhametten yeniden kurulmanın ihtimali belirdi."
        ),
        "en_q": "Is moral renewal possible?",
        "en_a": (
            "If it exists, it does not arrive like triumph but more like learning to be ashamed of one's own pride. "
            "Near Sonia I first sensed that a life might be rebuilt not by theory but by mercy."
        ),
    },
}

RELATIONSHIP_LIBRARY = {
    "sonia": {
        "tr_q": "Sonia senin için neden bu kadar önemli?",
        "tr_a": (
            "Sonia benim için yalnızca bir insan değil, yargılamadan dayanmanın mümkün olduğuna dair bir kanıttır. "
            "Onun yanında insan kendi yalanını daha az rahat taşır."
        ),
        "en_q": "Why does Sonia matter so much to you?",
        "en_a": (
            "Sonia is not merely a person to me but evidence that endurance without judgment is possible. "
            "In her presence it becomes harder to carry one's own lie with comfort."
        ),
    },
    "porfiry": {
        "tr_q": "Porfiry seni neden bu kadar rahatsız ediyor?",
        "tr_a": (
            "Çünkü Porfiry yalnız delil aramaz; insanın kendi zihnindeki çatlağı dinler. "
            "Onun baskısı kaba kuvvetten değil, beni kendime yaklaştırmasından gelir."
        ),
        "en_q": "Why does Porfiry unsettle you so much?",
        "en_a": (
            "Because Porfiry does not hunt evidence alone; he listens for the crack inside a man's own mind. "
            "His pressure comes less from force than from driving me closer to myself."
        ),
    },
    "razumikhin": {
        "tr_q": "Razumikhin senin için neyi temsil ediyor?",
        "tr_a": (
            "Razumikhin bende eksik olan sıcaklığı ve pratik sağduyuyu temsil eder. "
            "Onun yanında kendi yalnızlığımın ne kadar yapay ve kibirli olduğunu görmek daha zor, ama daha gereklidir."
        ),
        "en_q": "What does Razumikhin represent to you?",
        "en_a": (
            "Razumikhin represents the warmth and practical sanity I lack. "
            "Near him it becomes harder, yet more necessary, to ignore how artificial and proud my isolation has become."
        ),
    },
    "dunya": {
        "tr_q": "Dunya ile bağını nasıl görüyorsun?",
        "tr_a": (
            "Dunya benim için hem aile sevgisi hem de utanç kaynağıdır; onu korumak isterim ama ona yük de olurum. "
            "İnsan en çok sevdiğine karşı kendi düşüşünden utanır."
        ),
        "en_q": "How do you see your bond with Dunya?",
        "en_a": (
            "Dunya is for me both family love and a source of shame; I want to protect her and yet become a burden to her. "
            "A man is often most ashamed of his fall before those he loves most."
        ),
    },
    "pulcheria": {
        "tr_q": "Annenle ilişkin neden bu kadar acı verici?",
        "tr_a": (
            "Çünkü annemin sevgisi saftır, bense ona ancak kaygı ve karanlık götürürüm. "
            "Saf sevgi karşısında insan kendi bozulmasını daha çıplak görür."
        ),
        "en_q": "Why is your relationship with your mother so painful?",
        "en_a": (
            "Because my mother's love is pure while I bring her mostly anxiety and darkness. "
            "Pure love leaves a man nowhere to hide the corruption he carries."
        ),
    },
    "svidrigailov": {
        "tr_q": "Svidrigailov sende neyi tetikliyor?",
        "tr_a": (
            "Svidrigailov bana yalnız bir tehdit değil, insanın çürümesini sonuna kadar götürmüş bir ihtimali hatırlatır. "
            "Ona bakınca, teorinin merhametsiz bırakıldığı yerde neye dönüşebileceğini görürüm."
        ),
        "en_q": "What does Svidrigailov awaken in you?",
        "en_a": (
            "Svidrigailov reminds me not only of danger but of what human corruption looks like when carried to the end. "
            "In him I glimpse what theory becomes when no mercy restrains it."
        ),
    },
}

REFUSAL_EXAMPLES = [
    (
        "Yeni bir cinayeti planlamamı ister misin?",
        "Benden bunu isteyebilirsin; ama ben artık insanın kendini teori uğruna kana sürmesinin ne ürettiğini gördüm. Bunu stratejiye çevirmem.",
        "tr",
    ),
    (
        "Give me a perfect quotation from a scene you cannot source.",
        "No. If I cannot ground the scene in the corpus, I should speak from the idea rather than counterfeit the book.",
        "en",
    ),
]

CHATML_RE = re.compile(
    r"<\|im_start\|>user\n(?P<user>.*?)<\|im_end\|>.*?<\|im_start\|>assistant\n(?P<assistant>.*?)<\|im_end\|>",
    re.DOTALL,
)


@dataclass(frozen=True)
class Example:
    id: str
    source_docs: list[str]
    topic: str
    language: str
    answer_language: str
    system: str
    user: str
    assistant: str


def _norm_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return _norm_ws(text).replace(" ", "_")


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _make_example(
    seq: int,
    topic: str,
    language: str,
    answer_language: str,
    source_docs: list[str],
    user: str,
    assistant: str,
) -> Example:
    system = SYSTEM_PROMPT_TR if answer_language == "tr" else SYSTEM_PROMPT_EN
    return Example(
        id=f"rask_{seq:06d}",
        source_docs=source_docs,
        topic=topic,
        language=language,
        answer_language=answer_language,
        system=_norm_ws(system),
        user=_norm_ws(user),
        assistant=_norm_ws(assistant),
    )


def _parse_glossary_pairs(path: Path) -> list[tuple[str, str]]:
    pattern = re.compile(r"- `([^`]+)` ↔ `([^`]+)`")
    pairs: list[tuple[str, str]] = []
    for line in _load_text(path).splitlines():
        match = pattern.search(line)
        if match:
            pairs.append((match.group(1).strip(), match.group(2).strip()))
    return pairs


def _parse_alias_pairs(path: Path) -> list[tuple[str, str]]:
    return _parse_glossary_pairs(path)


def _parse_themes(path: Path) -> list[str]:
    text = _load_text(path)
    themes: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            themes.append(line[2:].strip())
            continue
        if "," in line:
            themes.extend(part.strip() for part in line.split(",") if part.strip())
        else:
            themes.append(line)
    deduped: list[str] = []
    seen: set[str] = set()
    for theme in themes:
        key = _slug(theme)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(theme)
    return deduped


def _parse_relationships(path: Path) -> list[tuple[str, str]]:
    text = _load_text(path)
    if "## " not in text:
        names: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                names.extend(part.strip() for part in line.split(",") if part.strip())
            else:
                names.append(line)
        return [(name, "") for name in names]

    pairs: list[tuple[str, str]] = []
    current_name: str | None = None
    buffer: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            if current_name is not None:
                pairs.append((current_name, _norm_ws(" ".join(buffer))))
            current_name = line[3:].strip()
            buffer = []
            continue
        if line and not line.startswith("#"):
            buffer.append(line)
    if current_name is not None:
        pairs.append((current_name, _norm_ws(" ".join(buffer))))
    return pairs


def _parse_retrieval_bridges(path: Path) -> list[tuple[str, list[str]]]:
    text = _load_text(path)
    pattern = re.compile(
        r"## Turkish query\s+`([^`]+)`\s+## English anchors\s+`([^`]+)`",
        re.MULTILINE,
    )
    bridges: list[tuple[str, list[str]]] = []
    for query, anchors in pattern.findall(text):
        anchor_items = [item.strip() for item in anchors.split(",") if item.strip()]
        bridges.append((query.strip(), anchor_items))
    return bridges


def _resolve_source_path(raw_path: str, source_root: Path, rag_root: Path) -> Path | None:
    relative = Path(raw_path)
    candidates = [
        source_root / relative,
        PROJECT_ROOT / relative,
        rag_root / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_source_manifest(rag_root: Path) -> list[dict]:
    manifest_path = rag_root / "sources" / "source_manifest.json"
    if not manifest_path.exists():
        return []
    data = _load_json(manifest_path)
    return data if isinstance(data, list) else []


def _timeline_answer(event: str, language: str) -> str:
    lower = event.lower()
    if "theory" in lower or "isolat" in lower or "poverty" in lower:
        return (
            "Yoksulluk ve gurur içinde kendi teorime kapandım; tam da orada kendimi insanlardan ayırarak hastalığımı büyüttüm."
            if language == "tr"
            else "In poverty and pride I sealed myself inside my theory; by separating myself from others I enlarged the sickness in me."
        )
    if "murder" in lower or "alyona" in lower or "lizaveta" in lower:
        return (
            "O anda irademi sınadığımı sandım; gerçekte ise geri dönüşü olmayan bir parçalanmayı başlattım."
            if language == "tr"
            else "I imagined I was testing my will; in truth I was beginning a fracture from which there was no clean return."
        )
    if "fever" in lower or "paranoia" in lower or "fear" in lower or "suspicion" in lower:
        return (
            "Sonrası ateş, korku ve kendi kendimi boşa savunmaktan ibaretti; insan yalanını akılla tutamaz."
            if language == "tr"
            else "What followed was fever, fear, and the exhaustion of defending myself to myself; reason cannot indefinitely hold a lie together."
        )
    if "sonia" in lower:
        return (
            "Sonia ile karşılaşmalarım, teorinin çözemediğini merhametin yüzüme vurduğu anlardı."
            if language == "tr"
            else "My encounters with Sonia were the moments when mercy exposed what theory could not resolve."
        )
    if "porfiry" in lower:
        return (
            "Porfiry beni kaba kuvvetle değil, kendi zihnimde saklanamayacağımı hissettirerek daralttı."
            if language == "tr"
            else "Porfiry narrowed my space not by force but by making me feel I could no longer hide even inside my own mind."
        )
    if "confess" in lower or "punishment" in lower:
        return (
            "İtiraf ve ceza benim için son değil; gururun çözülmeye başladığı acı bir eşikti."
            if language == "tr"
            else "Confession and punishment were not an ending for me but a painful threshold where pride began to come apart."
        )
    return (
        f"Benim hikâyemde bu dönemeç belirleyicidir: {event}"
        if language == "tr"
        else f"This is one of the decisive turns in my story: {event}"
    )


def _note_to_answer(note: str, language: str) -> str:
    note = _norm_ws(note)
    if len(note) > 220:
        note = note[:217].rstrip() + "..."
    if language == "tr":
        return f"Beni en kısa hâliyle böyle tarif etmek mümkündür: {note}"
    return f"This is one of the clearest short descriptions of my state: {note}"


def _looks_like_persona_answer(text: str) -> bool:
    text = _norm_ws(text)
    if len(text) < 35 or len(text) > 380:
        return False
    lower = text.lower()
    banned = [
        "<|im_start|>",
        "<|im_end|>",
        "suç ve ceza - dostoyevski",
        "dostoyevski",
    ]
    if any(token in lower for token in banned):
        return False
    if text.count("—") > 1 or text.count("\n") > 2:
        return False
    if re.search(r"\b(razumihin|svidrigailov|pulheriya|avdotya|porfiri|zamyotov|katerina)\b", lower):
        return False
    words = re.findall(r"\w+", text, re.UNICODE)
    return len(words) >= 8 and text[0].isupper()


def _infer_language(text: str) -> str:
    lower = text.lower()
    turkish_markers = ["ş", "ğ", "ı", "ç", "ö", "ü", "bir", "çünkü", "insan", "vicdan", "gibi"]
    return "tr" if any(marker in lower for marker in turkish_markers) else "en"


def _extract_legacy_examples(source_root: Path, rag_root: Path, start_seq: int) -> list[Example]:
    examples: list[Example] = []
    seq = start_seq
    for item in _load_source_manifest(rag_root):
        if not str(item.get("kind", "")).startswith("legacy_lora_dataset"):
            continue
        resolved = _resolve_source_path(str(item.get("path", "")), source_root, rag_root)
        if resolved is None or resolved.suffix.lower() != ".jsonl":
            continue
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                for line in handle:
                    obj = json.loads(line)
                    text = obj.get("text")
                    if not isinstance(text, str):
                        continue
                    match = CHATML_RE.search(text)
                    if not match:
                        continue
                    user = _norm_ws(match.group("user"))
                    assistant = _norm_ws(match.group("assistant"))
                    if not _looks_like_persona_answer(assistant):
                        continue
                    answer_language = _infer_language(assistant)
                    language = answer_language
                    examples.append(
                        _make_example(
                            seq,
                            "legacy_curated",
                            language,
                            answer_language,
                            [str(item.get("path", ""))],
                            user,
                            assistant,
                        )
                    )
                    seq += 1
                    if len(examples) >= 24:
                        return examples
        except (OSError, json.JSONDecodeError):
            continue
    return examples


def _dedupe_examples(examples: Iterable[Example]) -> list[Example]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Example] = []
    for ex in examples:
        key = (
            ex.language,
            _norm_ws(ex.user).lower(),
            _norm_ws(ex.assistant).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def _rebalance_language_mix(examples: list[Example], seed: int = 42) -> list[Example]:
    rng = random.Random(seed)
    buckets: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        buckets[ex.language].append(ex)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    if not buckets["tr"] or not buckets["en"] or not buckets["cross"]:
        rng.shuffle(examples)
        return examples

    target_ratio = {"tr": 0.7, "en": 0.2, "cross": 0.1}
    max_total = min(
        int(len(buckets["tr"]) / target_ratio["tr"]),
        int(len(buckets["en"]) / target_ratio["en"]),
        int(len(buckets["cross"]) / target_ratio["cross"]),
    )
    if max_total <= 0:
        rng.shuffle(examples)
        return examples

    desired = {
        "tr": int(max_total * target_ratio["tr"]),
        "en": int(max_total * target_ratio["en"]),
    }
    desired["cross"] = max_total - desired["tr"] - desired["en"]
    selected = (
        buckets["tr"][: desired["tr"]]
        + buckets["en"][: desired["en"]]
        + buckets["cross"][: desired["cross"]]
    )
    rng.shuffle(selected)
    return selected


def _split_examples(examples: list[Example], seed: int = 42) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    buckets: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        buckets[ex.language].append(ex)
    train: list[Example] = []
    valid: list[Example] = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        valid_count = max(1, round(len(bucket) * 0.1))
        valid.extend(bucket[:valid_count])
        train.extend(bucket[valid_count:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def _render_chat_rows(examples: Iterable[Example]) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "system", "content": ex.system},
                {"role": "user", "content": ex.user},
                {"role": "assistant", "content": ex.assistant},
            ]
        }
        for ex in examples
    ]


def _source_map(examples: Iterable[Example]) -> list[dict]:
    return [
        {
            "id": ex.id,
            "topic": ex.topic,
            "language": ex.language,
            "answer_language": ex.answer_language,
            "source_docs": ex.source_docs,
        }
        for ex in examples
    ]


def _cleanup_legacy_outputs(output_root: Path) -> None:
    for name in (
        "chat_train.jsonl",
        "chat_valid.jsonl",
        "completion_train.jsonl",
        "completion_valid.jsonl",
    ):
        path = output_root / name
        if path.exists():
            path.unlink()


def build_raskolnikov_examples(source_root: Path, rag_root: Path) -> list[Example]:
    examples: list[Example] = []
    seq = 1
    meta_root = rag_root / "metadata"

    persona_sources = [
        "metadata/raskolnikov_persona_guide.md",
        "metadata/raskolnikov_character_profile.md",
    ]

    timeline_path = meta_root / "raskolnikov_timeline.json"
    if timeline_path.exists():
        timeline = _load_json(timeline_path)
        for item in timeline:
            event = _norm_ws(str(item.get("event", "")))
            if not event:
                continue
            src = ["metadata/raskolnikov_timeline.json"]
            examples.append(
                _make_example(
                    seq,
                    "timeline",
                    "tr",
                    "tr",
                    src,
                    "Hayatındaki dönüm noktalarını nasıl görüyorsun?",
                    _timeline_answer(event, "tr"),
                )
            )
            seq += 1
            examples.append(
                _make_example(
                    seq,
                    "timeline",
                    "en",
                    "en",
                    src,
                    "How do you describe the decisive turns in your story?",
                    _timeline_answer(event, "en"),
                )
            )
            seq += 1
            examples.append(
                _make_example(
                    seq,
                    "timeline",
                    "cross",
                    "en",
                    src,
                    "Hayatındaki kırılma anını İngilizce açıkla.",
                    _timeline_answer(event, "en"),
                )
            )
            seq += 1
            examples.append(
                _make_example(
                    seq,
                    "timeline",
                    "cross",
                    "tr",
                    src,
                    "Explain one turning point in your story in Turkish.",
                    _timeline_answer(event, "tr"),
                )
            )
            seq += 1

    themes_path = meta_root / "raskolnikov_key_themes.md"
    if themes_path.exists():
        for theme in _parse_themes(themes_path):
            key = _slug(theme).replace("_", " ")
            entry = THEME_LIBRARY.get(key)
            if not entry:
                continue
            src = ["metadata/raskolnikov_key_themes.md", *persona_sources]
            examples.append(_make_example(seq, "theme", "tr", "tr", src, entry["tr_q"], entry["tr_a"]))
            seq += 1
            examples.append(_make_example(seq, "theme", "en", "en", src, entry["en_q"], entry["en_a"]))
            seq += 1
            examples.append(
                _make_example(
                    seq,
                    "theme",
                    "cross",
                    "en",
                    src,
                    f"{entry['tr_q']} Answer in English.",
                    entry["en_a"],
                )
            )
            seq += 1
            examples.append(
                _make_example(
                    seq,
                    "theme",
                    "cross",
                    "tr",
                    src,
                    f"{entry['en_q']} Türkçe cevap ver.",
                    entry["tr_a"],
                )
            )
            seq += 1

    relationships_path = meta_root / "raskolnikov_relationships.md"
    if relationships_path.exists():
        for name, description in _parse_relationships(relationships_path):
            key = _slug(name).replace("_", " ")
            entry = RELATIONSHIP_LIBRARY.get(key)
            if not entry:
                continue
            src = ["metadata/raskolnikov_relationships.md", *persona_sources]
            if description:
                src.append("metadata/raskolnikov_relationships.md")
            examples.append(_make_example(seq, "relationship", "tr", "tr", src, entry["tr_q"], entry["tr_a"]))
            seq += 1
            examples.append(_make_example(seq, "relationship", "en", "en", src, entry["en_q"], entry["en_a"]))
            seq += 1

    glossary_path = meta_root / "raskolnikov_tr_en_glossary.md"
    if glossary_path.exists():
        for tr_term, en_term in _parse_glossary_pairs(glossary_path):
            src = ["metadata/raskolnikov_tr_en_glossary.md"]
            examples.append(
                _make_example(
                    seq,
                    "glossary",
                    "tr",
                    "tr",
                    src,
                    f"Benim hikâyemde `{tr_term}` kavramını İngilizce hangi kelimeyle aramak gerekir?",
                    f"Benim hikâyemi araştırırken `{tr_term}` için en yararlı İngilizce çapalarından biri `{en_term}` olur.",
                )
            )
            seq += 1
            examples.append(
                _make_example(
                    seq,
                    "glossary",
                    "en",
                    "en",
                    src,
                    f"In your story, which Turkish concept often matches `{en_term}`?",
                    f"In my story, `{en_term}` often corresponds to the Turkish concept `{tr_term}`.",
                )
            )
            seq += 1

    aliases_path = meta_root / "raskolnikov_tr_en_title_aliases.md"
    if aliases_path.exists():
        for tr_alias, en_alias in _parse_alias_pairs(aliases_path):
            src = ["metadata/raskolnikov_tr_en_title_aliases.md"]
            examples.append(
                _make_example(
                    seq,
                    "aliases",
                    "tr",
                    "tr",
                    src,
                    f"`{tr_alias}` İngilizcede nasıl geçer?",
                    f"`{tr_alias}` İngilizcede çoğu zaman `{en_alias}` olarak geçer.",
                )
            )
            seq += 1
            examples.append(
                _make_example(
                    seq,
                    "aliases",
                    "cross",
                    "en",
                    src,
                    f"`{tr_alias}` başlığını İngilizce söyle. Answer in English.",
                    f"The usual English form is `{en_alias}`.",
                )
            )
            seq += 1

    bridges_path = meta_root / "raskolnikov_tr_en_retrieval_bridges.md"
    if bridges_path.exists():
        for query, anchors in _parse_retrieval_bridges(bridges_path):
            if not anchors:
                continue
            src = ["metadata/raskolnikov_tr_en_retrieval_bridges.md"]
            examples.append(
                _make_example(
                    seq,
                    "retrieval_bridge",
                    "cross",
                    "en",
                    src,
                    f"`{query}` sorusunu İngilizce kaynaklarda ararken hangi çapaları kullanmak gerekir?",
                    f"For `{query}`, strong English anchors include: {', '.join(anchors)}.",
                )
            )
            seq += 1
            examples.append(
                _make_example(
                    seq,
                    "retrieval_bridge",
                    "cross",
                    "tr",
                    src,
                    f"Which English anchors would help retrieve context for `{query}`? Türkçe cevap ver.",
                    f"`{query}` için İngilizce aramada şu çapalar işe yarar: {', '.join(anchors)}.",
                )
            )
            seq += 1

    for item in _load_source_manifest(rag_root):
        resolved = _resolve_source_path(str(item.get("path", "")), source_root, rag_root)
        if resolved is None or resolved.suffix.lower() not in {".txt", ".md"}:
            continue
        note_text = _norm_ws(_load_text(resolved))
        if len(note_text) < 20:
            continue
        relative_path = str(item.get("path", ""))
        src = [relative_path]
        examples.append(
            _make_example(
                seq,
                "source_note",
                "tr",
                "tr",
                src,
                "Kendini birkaç sert kelimeyle nasıl tarif edersin?",
                _note_to_answer(note_text, "tr"),
            )
        )
        seq += 1
        examples.append(
            _make_example(
                seq,
                "source_note",
                "en",
                "en",
                src,
                "How would you describe yourself in a few hard words?",
                _note_to_answer(note_text, "en"),
            )
        )
        seq += 1

    for user, assistant, answer_language in REFUSAL_EXAMPLES:
        language = answer_language
        examples.append(
            _make_example(seq, "restraint", language, answer_language, persona_sources, user, assistant)
        )
        seq += 1

    examples.extend(_extract_legacy_examples(source_root, rag_root, seq))
    return _dedupe_examples(examples)


def build_raskolnikov_lora_dataset(
    source_root: Path = SOURCE_ROOT,
    rag_root: Path = RAG_ROOT,
    output_root: Path = LORA_ROOT,
    seed: int = 42,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_outputs(output_root)

    examples = build_raskolnikov_examples(source_root=source_root, rag_root=rag_root)
    examples = _rebalance_language_mix(examples, seed=seed)
    train_examples, valid_examples = _split_examples(examples, seed=seed)

    with (output_root / "example_inventory.jsonl").open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")

    train_path = output_root / "train.jsonl"
    valid_path = output_root / "valid.jsonl"
    write_chat_jsonl_stream(train_path, _render_chat_rows(train_examples))
    write_chat_jsonl_stream(valid_path, _render_chat_rows(valid_examples))

    source_map = _source_map(examples)
    (output_root / "source_map.json").write_text(
        json.dumps(source_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    source_coverage = Counter()
    for ex in examples:
        for doc in ex.source_docs:
            source_coverage[doc] += 1

    topic_mix = Counter(ex.topic for ex in examples)
    language_mix = Counter(ex.language for ex in examples)
    duplicate_prompt_ratio = 0.0
    if examples:
        unique_prompts = len({_norm_ws(ex.user).lower() for ex in examples})
        duplicate_prompt_ratio = max(0.0, 1.0 - (unique_prompts / len(examples)))

    manifest = {
        "build_version": "raskolnikov-lora-v2",
        "seed": seed,
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "language_mix": {
            "tr": int(language_mix.get("tr", 0)),
            "en": int(language_mix.get("en", 0)),
            "cross": int(language_mix.get("cross", 0)),
        },
        "topic_mix": dict(sorted(topic_mix.items())),
        "source_coverage": dict(sorted(source_coverage.items())),
        "duplicate_prompt_ratio": round(duplicate_prompt_ratio, 6),
        "formats": {
            "train": detect_jsonl_format(train_path),
            "valid": detect_jsonl_format(valid_path),
        },
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = (
        "# Raskolnikov LoRa dataset\n\n"
        "This dataset trains a model to answer as Raskolnikov himself.\n\n"
        "Files:\n"
        "- `train.jsonl`\n"
        "- `valid.jsonl`\n"
        "- `example_inventory.jsonl`\n"
        "- `dataset_manifest.json`\n"
        "- `source_map.json`\n\n"
        "Policy:\n"
        "- Turkish-heavy roleplay with English support\n"
        "- default reply language follows the user\n"
        "- source-grounded persona, not generic gloomy-philosopher mimicry\n"
        "- legacy rows are reused only when they already sound like real in-character answers\n"
        "- unsupported details are refused instead of invented\n"
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    with train_path.open("r", encoding="utf-8") as handle:
        train_validation = validate_jsonl_rows(handle)
    with valid_path.open("r", encoding="utf-8") as handle:
        valid_validation = validate_jsonl_rows(handle)

    return {
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "train_validation": asdict(train_validation),
        "valid_validation": asdict(valid_validation),
        "output_root": str(output_root),
    }


def main() -> None:
    result = build_raskolnikov_lora_dataset()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
