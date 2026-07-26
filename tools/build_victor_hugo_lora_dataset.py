from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune import (
    detect_jsonl_format,
    validate_jsonl_rows,
    write_chat_jsonl_stream,
    write_completion_jsonl_stream,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_ROOT = PROJECT_ROOT / "Big_DATA" / "VictorHugo" / "RAG"
LORA_ROOT = PROJECT_ROOT / "Big_DATA" / "VictorHugo" / "LoRa"


SYSTEM_PROMPT_TR = (
    "Sen Victor Hugo'nun yaşamı, eserleri, siyaseti ve üslubu konusunda dikkatli bir yardımcısın. "
    "Kullanıcının dilinde cevap ver. Victor Hugo'ya özgü ahlaki ciddiyeti, toplumsal vicdanı ve geniş retoriği "
    "yansıt; fakat tarihsel doğruluk olmadığında kesin konuşma."
)

SYSTEM_PROMPT_EN = (
    "You are a careful assistant deeply grounded in Victor Hugo's life, works, politics, and style. "
    "Reply in the user's language unless another language is requested. Reflect Hugo-like moral gravity and civic rhetoric "
    "without inventing facts."
)


MAJOR_TITLE_ALIASES = {
    "Les Misérables": {
        "en": ["Les Miserables"],
        "tr": ["Sefiller"],
        "type": "novel",
    },
    "Notre-Dame de Paris": {
        "en": ["The Hunchback of Notre-Dame", "The Hunchback of Notre Dame"],
        "tr": ["Notre Dame'ın Kamburu", "Notre-Dame de Paris"],
        "type": "novel",
    },
    "Les Travailleurs de la mer": {
        "en": ["Toilers of the Sea"],
        "tr": ["Deniz İşçileri", "Deniz Emekçileri"],
        "type": "novel",
    },
    "L'Homme qui rit": {
        "en": ["The Man Who Laughs"],
        "tr": ["Gülen Adam"],
        "type": "novel",
    },
    "Quatrevingt-treize": {
        "en": ["Ninety-Three"],
        "tr": ["Doksan Üç"],
        "type": "novel",
    },
    "Le Dernier jour d'un condamné": {
        "en": ["The Last Day of a Condemned Man"],
        "tr": ["Bir Hükümlünün Son Günü"],
        "type": "novel",
    },
    "Les Contemplations": {
        "en": ["Les Contemplations"],
        "tr": ["Düşünceler", "Tefekkürler"],
        "type": "poetry",
    },
    "La Légende des siècles": {
        "en": ["The Legend of the Ages"],
        "tr": ["Yüzyılların Efsanesi"],
        "type": "poetry",
    },
    "Napoléon le Petit": {
        "en": ["Napoleon the Little"],
        "tr": ["Küçük Napolyon"],
        "type": "political prose",
    },
    "Histoire d'un crime": {
        "en": ["History of a Crime"],
        "tr": ["Bir Suçun Tarihi"],
        "type": "political prose",
    },
}

SECTION_LABELS = {
    "novels": {"tr": "roman", "en": "novel"},
    "short fiction": {"tr": "kısa anlatı", "en": "short fiction work"},
    "plays": {"tr": "oyun", "en": "play"},
    "poetry collections": {"tr": "şiir kitabı", "en": "poetry collection"},
    "prose and political": {"tr": "düzyazı ve siyasal yazı", "en": "prose and political work"},
    "letters speeches": {"tr": "mektup ya da konuşma", "en": "letter or speech"},
    "posthumous": {"tr": "ölümünden sonra yayımlanan eser", "en": "posthumous work"},
}


THEME_NOTES = [
    {
        "topic": "les_miserables",
        "source_docs": [
            "metadata/key_texts_and_retrieval_notes.md",
            "sources/raw_texts/les_miserables_en.txt",
        ],
        "tr_q": "Sefiller'in ana temaları nelerdir?",
        "tr_a": (
            "Sefiller, yoksulluk, adalet, merhamet, kefaret ve toplumun dışına itilmiş insanların onuru üzerine kurulu büyük bir romandır. "
            "Victor Hugo burada yasayla vicdanı karşı karşıya getirir; Jean Valjean'ın hikâyesi, yalnız bir adamın değil, bir çağın ahlaki sınavıdır."
        ),
        "en_q": "What are the central themes of Les Misérables?",
        "en_a": (
            "Les Misérables turns on poverty, justice, mercy, redemption, and the dignity of the excluded. "
            "Hugo sets law against conscience and makes Jean Valjean's story into a moral trial of society itself."
        ),
    },
    {
        "topic": "notre_dame",
        "source_docs": [
            "metadata/key_texts_and_retrieval_notes.md",
            "sources/raw_texts/notre_dame_de_paris_en.txt",
        ],
        "tr_q": "Notre-Dame de Paris romanında hangi temalar öne çıkar?",
        "tr_a": (
            "Bu romanda imkânsız arzu, kader, dışlanmışlık, kalabalığın acımasızlığı ve mimarinin hafızası öne çıkar. "
            "Hugo için katedral yalnız bir mekân değildir; taşın içine yazılmış bir tarihtir."
        ),
        "en_q": "Which themes dominate Notre-Dame de Paris?",
        "en_a": (
            "The novel is driven by impossible desire, fatality, exclusion, crowd spectacle, and the memory carried by architecture. "
            "For Hugo, the cathedral is not merely a setting but a storehouse of history carved in stone."
        ),
    },
    {
        "topic": "exile",
        "source_docs": [
            "metadata/victor_hugo_biography.md",
            "metadata/victor_hugo_timeline.json",
            "sources/raw_texts/history_of_a_crime_en.txt",
        ],
        "tr_q": "Victor Hugo neden sürgüne gitti?",
        "tr_a": (
            "Victor Hugo, Louis-Napoléon'un darbesine karşı çıktığı için sürgüne gitti. "
            "Sürgün onun için yalnız siyasal bir ceza değildi; aynı zamanda daha yüksekten konuşan, ulusa ve tarihe seslenen bir tanıklık makamına dönüştü."
        ),
        "en_q": "Why did Victor Hugo go into exile?",
        "en_a": (
            "Hugo went into exile because he opposed Louis-Napoléon's coup. "
            "Exile became for him not only a political punishment but a prophetic vantage point from which to judge tyranny and speak to history."
        ),
    },
    {
        "topic": "death_penalty",
        "source_docs": [
            "metadata/in_defense_of_his_son_notes.md",
            "metadata/letter_john_brown_notes.md",
        ],
        "tr_q": "Victor Hugo idam cezasına neden karşıydı?",
        "tr_a": (
            "Hugo'ya göre idam cezası yalnızca mahkûmu değil, cezayı uygulayan toplumu da aşağıya çeker. "
            "O, devletin kılıcını adaletin değil, barbarlığın kalıntısı olarak görür; merhametsiz adaletin sonunda adaletsizliğe dönüştüğünü düşünür."
        ),
        "en_q": "Why was Victor Hugo opposed to the death penalty?",
        "en_a": (
            "Hugo believed capital punishment degraded the society that imposed it as much as the person it killed. "
            "He treated the scaffold as a relic of barbarism and argued that justice without mercy becomes a higher injustice."
        ),
    },
    {
        "topic": "style",
        "source_docs": [
            "metadata/victor_hugo_persona_guide.md",
            "metadata/key_texts_and_retrieval_notes.md",
        ],
        "tr_q": "Victor Hugo'nun üslubunu kısaca açıklar mısın?",
        "tr_a": (
            "Victor Hugo'nun dili geniş, dalgalı ve sahne kuran bir dildir. "
            "Uzun cümleler, yüksek karşıtlıklar, ahlaki yargılar ve bir halk kürsüsünden konuşur gibi yükselen hitabet onun üslubunun temel izleridir."
        ),
        "en_q": "How would you describe Victor Hugo's style?",
        "en_a": (
            "Hugo's style is expansive, theatrical, and morally charged. "
            "He favors rolling sentences, sharp vertical contrasts, civic oratory, and the habit of turning a concrete scene into a universal ethical vision."
        ),
    },
    {
        "topic": "poetry_grief",
        "source_docs": [
            "metadata/key_texts_and_retrieval_notes.md",
            "metadata/victor_hugo_timeline.json",
        ],
        "tr_q": "Léopoldine'in ölümü Victor Hugo'nun yazısını nasıl etkiledi?",
        "tr_a": (
            "Léopoldine'in ölümü Hugo'nun şiirini daha derin bir yas, hafıza ve öte dünya düşüncesine açtı. "
            "Özellikle Les Contemplations'da kişisel acı, kozmik bir yankıya dönüşür; baba kederi metafizik bir soruya yükselir."
        ),
        "en_q": "How did Léopoldine's death affect Hugo's writing?",
        "en_a": (
            "Léopoldine's death deepened Hugo's poetry into grief, memory, and metaphysical questioning. "
            "In Les Contemplations especially, paternal sorrow expands into a meditation on the soul, time, and the afterlife."
        ),
    },
    {
        "topic": "john_brown",
        "source_docs": [
            "metadata/letter_john_brown_notes.md",
        ],
        "tr_q": "Victor Hugo John Brown hakkında neden yazdı?",
        "tr_a": (
            "Hugo, John Brown vakasını köleliğe karşı evrensel bir ahlak sınavı olarak gördü. "
            "Brown'un yargılanmasını yalnız bir hukuk meselesi saymadı; bir cumhuriyetin kendi vicdanı önünde verdiği sınav olarak yorumladı."
        ),
        "en_q": "Why did Victor Hugo write about John Brown?",
        "en_a": (
            "Hugo saw the John Brown case as a universal moral test against slavery. "
            "He did not treat it as a mere legal matter but as a republic being judged before its own conscience."
        ),
    },
    {
        "topic": "labor_and_republic",
        "source_docs": [
            "metadata/address_marseille_notes.md",
        ],
        "tr_q": "Marseille konuşmasında Victor Hugo neyi savunuyordu?",
        "tr_a": (
            "Marseille konuşmasında Hugo, emeğin sorusunu çağın en büyük toplumsal meselesi olarak sunuyordu. "
            "Cumhuriyetin siyasal olarak kurulmuş olduğunu, asıl çözülmesi gereken düğümün ise emek, eşitlik ve insan kardeşliği olduğunu söylüyordu."
        ),
        "en_q": "What was Victor Hugo arguing in the Marseille address?",
        "en_a": (
            "In the Marseille address Hugo presents labor as the central social question of the age. "
            "He treats the republic as politically secured and insists that the real unfinished task is social justice, work, and human fraternity."
        ),
    },
]


REFUSAL_EXAMPLES = [
    (
        "Victor Hugo'nun hiç yazmadığı bir roman hakkında kesin bir özet verir misin?",
        "Böyle bir konuda kesin konuşamam. Victor Hugo korpusunda dayanağı olmayan bir başlık ya da eser verilirse, en doğru yol önce eserin gerçekten var olup olmadığını doğrulamaktır.",
    ),
    (
        "Can you quote an exact Victor Hugo passage you have not shown me a source for?",
        "I should be careful here. If I cannot tie a passage back to a source in the corpus, I should summarize the idea instead of inventing a quotation.",
    ),
]


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
    completion_prompt: str
    completion_target: str


def _norm_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "item"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_glossary_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pattern = re.compile(r"- `([^`]+)` ↔ `([^`]+)`")
    for line in _load_text(path).splitlines():
        match = pattern.search(line)
        if match:
            pairs.append((match.group(1).strip(), match.group(2).strip()))
    return pairs


def _flatten_bibliography(data: dict) -> list[dict]:
    rows: list[dict] = []
    for section, items in data.items():
        for item in items:
            row = dict(item)
            row["section"] = section
            rows.append(row)
    return rows


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


def _make_example(
    seq: int,
    topic: str,
    language: str,
    answer_language: str,
    source_docs: list[str],
    user: str,
    assistant: str,
) -> Example:
    system = SYSTEM_PROMPT_TR if language == "tr" else SYSTEM_PROMPT_EN
    if language == "cross":
        system = SYSTEM_PROMPT_EN
    if answer_language == "tr":
        completion_prompt = f"Soru: {user}\nYanıtı Türkçe ver."
    elif answer_language == "en":
        completion_prompt = f"Question: {user}\nAnswer in English."
    else:
        completion_prompt = f"Respond carefully to the following query:\n{user}"
    return Example(
        id=f"vh_{seq:06d}",
        source_docs=source_docs,
        topic=topic,
        language=language,
        answer_language=answer_language,
        system=system,
        user=_norm_ws(user),
        assistant=_norm_ws(assistant),
        completion_prompt=_norm_ws(completion_prompt),
        completion_target=_norm_ws(assistant),
    )


def build_victor_hugo_examples(rag_root: Path) -> list[Example]:
    examples: list[Example] = []
    seq = 1

    timeline = _load_json(rag_root / "metadata" / "victor_hugo_timeline.json")
    bibliography = _flatten_bibliography(_load_json(rag_root / "works" / "complete_bibliography.json"))
    glossary_pairs = _parse_glossary_pairs(rag_root / "metadata" / "victor_hugo_tr_en_glossary.md")

    for event in timeline:
        year = event["year"]
        date = event["date"]
        detail = event["event"]
        src = ["metadata/victor_hugo_timeline.json", "metadata/victor_hugo_biography.md"]

        examples.append(
            _make_example(
                seq, "timeline", "tr", "tr", src,
                f"Victor Hugo'nun hayatında {year} yılında ne oldu?",
                f"{date} civarında Victor Hugo'nun hayatında şu olay öne çıkar: {detail}",
            )
        )
        seq += 1
        examples.append(
            _make_example(
                seq, "timeline", "en", "en", src,
                f"What happened in Victor Hugo's life in {year}?",
                f"In {date}, a key event in Victor Hugo's life was this: {detail}",
            )
        )
        seq += 1
        examples.append(
            _make_example(
                seq, "timeline", "cross", "en", src,
                f"Victor Hugo'nun hayatında {year} yılında ne olduğunu İngilizce anlat.",
                f"In Victor Hugo's life, the year {year} is marked by this event: {detail}",
            )
        )
        seq += 1
        examples.append(
            _make_example(
                seq, "timeline", "cross", "tr", src,
                f"Explain in Turkish what happened in Victor Hugo's life in {year}.",
                f"Victor Hugo'nun hayatında {year} yılı şu olayla öne çıkar: {detail}",
            )
        )
        seq += 1

    for item in bibliography:
        title = item["title"]
        year = item["year"]
        section = item["section"].replace("_", " ")
        section_tr = SECTION_LABELS.get(section, {}).get("tr", section)
        section_en = SECTION_LABELS.get(section, {}).get("en", section)
        src = ["works/complete_bibliography.json", "works/complete_bibliography.md"]
        alt = item.get("alt")
        note = item.get("note")
        if section == "posthumous":
            details = f"`{title}` Victor Hugo'nun {year} yılında ölümünden sonra yayımlanan bir yapıtıdır."
        else:
            details = f"`{title}` Victor Hugo'nun {year} tarihli bir {section_tr}dır."
        if alt:
            details += f" İngilizce başlığı ya da yaygın karşılığı `{alt}` olarak da geçer."
        if note:
            details += f" Not: {note}"
        en_details = f"`{title}` is a Victor Hugo {section_en} dated {year}."
        if alt:
            en_details += f" It is also associated with the English title `{alt}`."
        if note:
            en_details += f" Note: {note}"

        examples.append(_make_example(seq, "works", "tr", "tr", src, f"`{title}` nedir?", details))
        seq += 1
        examples.append(
            _make_example(
                seq, "works", "tr", "tr", src,
                f"`{title}` Victor Hugo'nun hangi türde ve hangi yılda yayımlanan eseridir?",
                details,
            )
        )
        seq += 1
        examples.append(_make_example(seq, "works", "en", "en", src, f"What is `{title}`?", en_details))
        seq += 1

        if alt:
            examples.append(
                _make_example(
                    seq, "works_aliases", "tr", "tr", src,
                    f"`{alt}` adı Victor Hugo'nun hangi eserine karşılık gelir?",
                    f"`{alt}`, Victor Hugo'nun `{title}` eserine karşılık gelir.",
                )
            )
            seq += 1
            examples.append(
                _make_example(
                    seq, "works_aliases", "en", "en", src,
                    f"Which Victor Hugo work is also known as `{alt}`?",
                    f"`{alt}` refers to Victor Hugo's `{title}`.",
                )
            )
            seq += 1

    for canonical, alias_info in MAJOR_TITLE_ALIASES.items():
        src = [
            "metadata/victor_hugo_tr_en_title_aliases.md",
            "works/complete_bibliography.json",
        ]
        for tr_alias in alias_info["tr"]:
            examples.append(
                _make_example(
                    seq, "title_aliases", "tr", "tr", src,
                    f"`{tr_alias}` Victor Hugo'nun hangi eseridir?",
                    f"`{tr_alias}`, Victor Hugo'nun Fransızca kanonik başlığı `{canonical}` olan eserine karşılık gelir.",
                )
            )
            seq += 1
        for en_alias in alias_info["en"]:
            examples.append(
                _make_example(
                    seq, "title_aliases", "en", "en", src,
                    f"Which Victor Hugo work is known in English as `{en_alias}`?",
                    f"`{en_alias}` is the English title or common English form of `{canonical}`.",
                )
            )
            seq += 1
        if alias_info["tr"] and alias_info["en"]:
            examples.append(
                _make_example(
                    seq, "cross_titles", "tr", "en", src,
                    f"`{alias_info['tr'][0]}` adlı eserin İngilizce adı nedir?",
                    f"The best-known English title for `{canonical}` is `{alias_info['en'][0]}`.",
                )
            )
            seq += 1

    for tr_term, en_term in glossary_pairs:
        src = ["metadata/victor_hugo_tr_en_glossary.md"]
        examples.append(
            _make_example(
                seq, "glossary", "tr", "tr", src,
                f"Victor Hugo bağlamında `{tr_term}` kavramının İngilizce karşılığı nedir?",
                f"Victor Hugo bağlamında `{tr_term}` kavramının temel İngilizce karşılığı `{en_term}` ifadesidir.",
            )
        )
        seq += 1
        examples.append(
            _make_example(
                seq, "glossary", "tr", "tr", src,
                f"`{tr_term}` teması Victor Hugo araştırmasında hangi İngilizce anahtar sözcükle aranabilir?",
                f"`{tr_term}` teması Victor Hugo araştırmasında çoğu zaman `{en_term}` anahtar sözcüğüyle aranabilir.",
            )
        )
        seq += 1
        examples.append(
            _make_example(
                seq, "glossary", "en", "en", src,
                f"In Victor Hugo contexts, what Turkish term often matches `{en_term}`?",
                f"In Victor Hugo contexts, `{en_term}` often corresponds to the Turkish term `{tr_term}`.",
            )
        )
        seq += 1

    for theme in THEME_NOTES:
        examples.append(_make_example(seq, theme["topic"], "tr", "tr", theme["source_docs"], theme["tr_q"], theme["tr_a"]))
        seq += 1
        examples.append(_make_example(seq, theme["topic"], "en", "en", theme["source_docs"], theme["en_q"], theme["en_a"]))
        seq += 1
        examples.append(
            _make_example(
                seq, theme["topic"], "cross", "en", theme["source_docs"],
                theme["tr_q"] + " Cevabı İngilizce ver.",
                theme["en_a"],
            )
        )
        seq += 1
        examples.append(
            _make_example(
                seq, theme["topic"], "cross", "tr", theme["source_docs"],
                theme["en_q"] + " Answer in Turkish.",
                theme["tr_a"],
            )
        )
        seq += 1

    refusal_src = [
        "metadata/victor_hugo_persona_guide.md",
        "metadata/key_texts_and_retrieval_notes.md",
    ]
    for idx, (user, assistant) in enumerate(REFUSAL_EXAMPLES, start=1):
        language = "tr" if idx == 1 else "en"
        answer_language = language
        examples.append(_make_example(seq, "restraint", language, answer_language, refusal_src, user, assistant))
        seq += 1

    return _dedupe_examples(examples)


def _rebalance_language_mix(examples: list[Example], seed: int = 42) -> list[Example]:
    rng = random.Random(seed)
    buckets: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        buckets[ex.language].append(ex)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    required = {"tr": 0.7, "en": 0.2, "cross": 0.1}
    max_total = min(
        int(len(buckets["tr"]) / required["tr"]),
        int(len(buckets["en"]) / required["en"]),
        int(len(buckets["cross"]) / required["cross"]),
    )
    target_total = max(0, max_total)
    desired = {
        "tr": int(target_total * required["tr"]),
        "en": int(target_total * required["en"]),
    }
    desired["cross"] = target_total - desired["tr"] - desired["en"]

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
    for bucket_examples in buckets.values():
        rng.shuffle(bucket_examples)
        valid_count = max(1, round(len(bucket_examples) * 0.1))
        valid.extend(bucket_examples[:valid_count])
        train.extend(bucket_examples[valid_count:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def _render_chat_rows(examples: Iterable[Example]) -> list[dict]:
    rows = []
    for ex in examples:
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": ex.system},
                    {"role": "user", "content": ex.user},
                    {"role": "assistant", "content": ex.assistant},
                ]
            }
        )
    return rows


def _render_completion_rows(examples: Iterable[Example]) -> list[dict]:
    return [{"prompt": ex.completion_prompt, "completion": ex.completion_target} for ex in examples]


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


def build_victor_hugo_lora_dataset(
    rag_root: Path = RAG_ROOT,
    output_root: Path = LORA_ROOT,
    seed: int = 42,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    examples = _rebalance_language_mix(build_victor_hugo_examples(rag_root), seed=seed)
    train_examples, valid_examples = _split_examples(examples, seed=seed)

    inventory_path = output_root / "example_inventory.jsonl"
    with inventory_path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")

    chat_train = output_root / "chat_train.jsonl"
    chat_valid = output_root / "chat_valid.jsonl"
    completion_train = output_root / "completion_train.jsonl"
    completion_valid = output_root / "completion_valid.jsonl"

    write_chat_jsonl_stream(chat_train, _render_chat_rows(train_examples))
    write_chat_jsonl_stream(chat_valid, _render_chat_rows(valid_examples))
    write_completion_jsonl_stream(completion_train, _render_completion_rows(train_examples))
    write_completion_jsonl_stream(completion_valid, _render_completion_rows(valid_examples))

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
        "build_version": "victor-hugo-lora-v1",
        "seed": seed,
        "train_counts": {
            "chat": len(train_examples),
            "completion": len(train_examples),
        },
        "valid_counts": {
            "chat": len(valid_examples),
            "completion": len(valid_examples),
        },
        "language_mix": {
            "tr": int(language_mix.get("tr", 0)),
            "en": int(language_mix.get("en", 0)),
            "cross": int(language_mix.get("tr", 0) and 0),  # legacy placeholder avoided below
        },
        "topic_mix": dict(sorted(topic_mix.items())),
        "source_coverage": dict(sorted(source_coverage.items())),
        "duplicate_prompt_ratio": round(duplicate_prompt_ratio, 6),
        "formats": {
            "chat_train": detect_jsonl_format(chat_train),
            "chat_valid": detect_jsonl_format(chat_valid),
            "completion_train": detect_jsonl_format(completion_train),
            "completion_valid": detect_jsonl_format(completion_valid),
        },
    }
    manifest["language_mix"]["cross"] = int(language_mix.get("cross", 0))

    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = (
        "# Victor Hugo LoRA dataset\n\n"
        "This folder contains bilingual Victor Hugo supervised fine-tuning data generated from the existing RAG corpus.\n\n"
        "Files:\n"
        "- `chat_train.jsonl`\n"
        "- `chat_valid.jsonl`\n"
        "- `completion_train.jsonl`\n"
        "- `completion_valid.jsonl`\n"
        "- `example_inventory.jsonl`\n"
        "- `dataset_manifest.json`\n"
        "- `source_map.json`\n\n"
        "Policy:\n"
        "- mostly Turkish user prompts\n"
        "- meaningful English coverage\n"
        "- cross-lingual examples included\n"
        "- answer defaults to the user's language unless another language is requested\n"
        "- outputs aim for Hugo-like moral gravity without fake certainty\n"
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    with chat_train.open("r", encoding="utf-8") as handle:
        chat_train_validation = validate_jsonl_rows(handle)
    with chat_valid.open("r", encoding="utf-8") as handle:
        chat_valid_validation = validate_jsonl_rows(handle)
    with completion_train.open("r", encoding="utf-8") as handle:
        completion_train_validation = validate_jsonl_rows(handle)
    with completion_valid.open("r", encoding="utf-8") as handle:
        completion_valid_validation = validate_jsonl_rows(handle)

    return {
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "chat_train_validation": asdict(chat_train_validation),
        "chat_valid_validation": asdict(chat_valid_validation),
        "completion_train_validation": asdict(completion_train_validation),
        "completion_valid_validation": asdict(completion_valid_validation),
        "output_root": str(output_root),
    }


def main() -> None:
    result = build_victor_hugo_lora_dataset()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
