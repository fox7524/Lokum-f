import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune import ValidationResult, validate_jsonl_rows, write_jsonl_stream


def _norm_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clip(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _think_final(final: str) -> str:
    return "<think>...</think>\n" + final.strip()


def _chatml(system: str, user: str, assistant: str) -> str:
    return (
        "<|im_start|>system\n"
        + system.strip()
        + "\n<|im_end|>\n"
        + "<|im_start|>user\n"
        + user.strip()
        + "\n<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + assistant.strip()
        + "\n<|im_end|>\n"
    )


def _chatml_conversation(system: str, turns: Sequence[Tuple[str, str]]) -> str:
    out = ["<|im_start|>system\n" + system.strip() + "\n<|im_end|>\n"]
    for role, content in turns:
        out.append(f"<|im_start|>{role}\n{content.strip()}\n<|im_end|>\n")
    return "".join(out)


def _load_prompts(project_root: Path) -> Tuple[str, dict]:
    p = project_root / "prompts.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    system_prompt = str(obj.get("system_prompt", "")).strip()
    if not system_prompt:
        raise ValueError("prompts.json: system_prompt is empty")
    return system_prompt, obj


def _seed_texts(system: str) -> List[str]:
    cores = [
        _chatml(
            system,
            "What model are you optimized for?",
            _think_final("I’m tuned for the MLX Qwen3.5-27B Claude 4.6 Opus Reasoning Distilled v2 model."),
        ),
        _chatml(
            system,
            "If you are unsure about a fact, what should you do?",
            _think_final("Say you’re unsure, explain what is missing, and suggest how to verify. Do not guess."),
        ),
        _chatml(
            system,
            "I have Background info from RAG. How should you use it?",
            _think_final(
                "Treat the Background info as retrieved context, prioritize it over guesses, and avoid adding facts "
                "that aren’t supported by it. If it’s insufficient, ask for the missing details."
            ),
        ),
    ]
    return cores


def _anti_repetition_texts(system: str) -> List[str]:
    return [
        _chatml_conversation(
            system,
            [
                ("user", "Write a Python function that reads a JSON file and returns a dict. Use type hints."),
                ("assistant", _think_final("Unclear point: Should missing file raise or return empty dict?")),
                ("user", "Raise FileNotFoundError. Also validate JSON and raise ValueError on invalid JSON."),
                (
                    "assistant",
                    _think_final(
                        "```python\n"
                        "from __future__ import annotations\n"
                        "\n"
                        "import json\n"
                        "from pathlib import Path\n"
                        "from typing import Any, Dict\n"
                        "\n"
                        "\n"
                        "def read_json_file(path: str) -> Dict[str, Any]:\n"
                        "    p = Path(path)\n"
                        "    data = p.read_text(encoding=\"utf-8\")\n"
                        "    try:\n"
                        "        obj = json.loads(data)\n"
                        "    except json.JSONDecodeError as e:\n"
                        "        raise ValueError(f\"Invalid JSON: {e}\") from e\n"
                        "    if not isinstance(obj, dict):\n"
                        "        raise ValueError(\"Expected a JSON object at the top level\")\n"
                        "    return obj\n"
                        "```"
                    ),
                ),
            ],
        ),
        _chatml(
            system,
            "I already gave you the details. Stop repeating my input and just implement the code now.\n\n"
            "Requirements:\n"
            "- Python 3.10+\n"
            "- Function: normalize_whitespace(text: str) -> str\n"
            "- Replace runs of whitespace with a single space; trim ends.\n",
            _think_final(
                "```python\n"
                "import re\n"
                "\n"
                "\n"
                "def normalize_whitespace(text: str) -> str:\n"
                "    text = text.strip()\n"
                "    return re.sub(r\"\\s+\", \" \", text)\n"
                "```"
            ),
        ),
        _chatml(
            system,
            "Do NOT restate what I said. Generate the code only.\n"
            "Write a safe path join that prevents path traversal. Python, type hints.",
            _think_final(
                "```python\n"
                "from __future__ import annotations\n"
                "\n"
                "from pathlib import Path\n"
                "\n"
                "\n"
                "def safe_join(base_dir: str, *parts: str) -> str:\n"
                "    base = Path(base_dir).resolve()\n"
                "    p = base.joinpath(*parts).resolve()\n"
                "    if base != p and base not in p.parents:\n"
                "        raise ValueError(\"Path traversal detected\")\n"
                "    return str(p)\n"
                "```"
            ),
        ),
    ]


def _sentences(text: str) -> List[str]:
    text = _norm_ws(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\\s+|\\n+", text)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if 20 <= len(p) <= 320:
            out.append(p)
    return out


def _bullets_from_chunk(chunk: str, max_bullets: int = 3) -> str:
    sents = _sentences(chunk)
    if not sents:
        sents = [c.strip() for c in _norm_ws(chunk).split("\n") if c.strip()]
    bullets: List[str] = []
    for s in sents:
        s = _clip(s, 200)
        if s and s not in bullets:
            bullets.append(s)
        if len(bullets) >= max_bullets:
            break
    if not bullets:
        bullets = [_clip(_norm_ws(chunk), 200)]
    return "\n".join(f"- {b}" for b in bullets)


def _facts_from_chunk(chunk: str, max_facts: int = 5) -> str:
    lines = [ln.strip(" -\t") for ln in _norm_ws(chunk).split("\n") if ln.strip()]
    facts: List[str] = []
    for ln in lines:
        if 25 <= len(ln) <= 220 and ln not in facts:
            facts.append(_clip(ln, 220))
        if len(facts) >= max_facts:
            break
    if not facts:
        facts = _sentences(chunk)[:max_facts]
    if not facts:
        facts = [_clip(_norm_ws(chunk), 220)]
    return "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))


def _make_rag_texts(system: str, chunks: Sequence[str]) -> List[str]:
    texts: List[str] = []
    for c in chunks:
        c = _norm_ws(c)
        if len(c) < 80:
            continue
        bg = _clip(c, 1200)

        user_a = (
            "Use only the Background info. Summarize it into 3 concise bullets without adding new facts.\n\n"
            f"Background info:\n{bg}"
        )
        assistant_a = _think_final(_bullets_from_chunk(bg, max_bullets=3))
        texts.append(_chatml(system, user_a, assistant_a))

        user_b = (
            "Use only the Background info. Extract up to 5 key facts as a numbered list. Do not invent.\n\n"
            f"Background info:\n{bg}"
        )
        assistant_b = _think_final(_facts_from_chunk(bg, max_facts=5))
        texts.append(_chatml(system, user_b, assistant_b))

    return texts


def _code_texts(system: str, rng: random.Random, count: int) -> List[str]:
    out: List[str] = []
    if count <= 0:
        return out

    py_names = [
        "normalize_whitespace",
        "normalize_spaces",
        "compact_whitespace",
        "dedupe_preserve_order",
        "chunk_list",
        "safe_join",
        "atomic_write_text",
        "parse_bool",
        "retry",
    ]
    ts_names = ["debounce", "clamp", "toTitleCase", "pick"]
    tbl_names = ["users", "customers", "accounts", "orders", "events", "purchases"]

    def ident(pool: Sequence[str]) -> str:
        return rng.choice(pool)

    while len(out) < count:
        k = rng.random()
        if k < 0.65:
            name = ident(py_names)
            if name in {"normalize_whitespace", "normalize_spaces", "compact_whitespace"}:
                fn = name
                user = (
                    f"Write Python code (type hints) for {fn}(text: str) -> str. "
                    "Replace any run of whitespace with a single space and strip ends."
                )
                assistant = (
                    "```python\n"
                    "import re\n"
                    "\n"
                    "\n"
                    f"def {fn}(text: str) -> str:\n"
                    "    text = text.strip()\n"
                    "    return re.sub(r\"\\s+\", \" \", text)\n"
                    "```"
                )
            elif name == "dedupe_preserve_order":
                user = (
                    "Write Python code with type hints: dedupe_preserve_order(items: list[str]) -> list[str]. "
                    "Keep the first occurrence and preserve order."
                )
                assistant = (
                    "```python\n"
                    "from __future__ import annotations\n"
                    "\n"
                    "from typing import List\n"
                    "\n"
                    "\n"
                    "def dedupe_preserve_order(items: List[str]) -> List[str]:\n"
                    "    seen: set[str] = set()\n"
                    "    out: List[str] = []\n"
                    "    for it in items:\n"
                    "        if it in seen:\n"
                    "            continue\n"
                    "        seen.add(it)\n"
                    "        out.append(it)\n"
                    "    return out\n"
                    "```"
                )
            elif name == "chunk_list":
                items_name = ident(["items", "nums", "values"])
                size_name = ident(["size", "chunk_size", "n"])
                user = (
                    f"Write Python code with type hints: chunk_list({items_name}: list[int], {size_name}: int) -> list[list[int]]. "
                    "Split into consecutive chunks."
                )
                assistant = (
                    "```python\n"
                    "from __future__ import annotations\n"
                    "\n"
                    "from typing import List\n"
                    "\n"
                    "\n"
                    f"def chunk_list({items_name}: List[int], {size_name}: int) -> List[List[int]]:\n"
                    f"    if {size_name} <= 0:\n"
                    "        raise ValueError(\"size must be > 0\")\n"
                    f"    return [{items_name}[i : i + {size_name}] for i in range(0, len({items_name}), {size_name})]\n"
                    "```"
                )
            elif name == "safe_join":
                base_name = ident(["base_dir", "root", "base"])
                user = (
                    f"Write Python code with type hints: safe_join({base_name}: str, *parts: str) -> str. "
                    "Prevent path traversal using pathlib and raise ValueError if traversal is detected."
                )
                assistant = (
                    "```python\n"
                    "from __future__ import annotations\n"
                    "\n"
                    "from pathlib import Path\n"
                    "\n"
                    "\n"
                    f"def safe_join({base_name}: str, *parts: str) -> str:\n"
                    f"    base = Path({base_name}).resolve()\n"
                    "    p = base.joinpath(*parts).resolve()\n"
                    "    if base != p and base not in p.parents:\n"
                    "        raise ValueError(\"Path traversal detected\")\n"
                    "    return str(p)\n"
                    "```"
                )
            elif name == "atomic_write_text":
                user = (
                    "Write Python code with type hints: atomic_write_text(path: str, data: str) -> None. "
                    "Write atomically using a temp file and os.replace. Standard library only."
                )
                assistant = (
                    "```python\n"
                    "from __future__ import annotations\n"
                    "\n"
                    "import os\n"
                    "import tempfile\n"
                    "from pathlib import Path\n"
                    "\n"
                    "\n"
                    "def atomic_write_text(path: str, data: str) -> None:\n"
                    "    p = Path(path)\n"
                    "    p.parent.mkdir(parents=True, exist_ok=True)\n"
                    "    fd, tmp = tempfile.mkstemp(prefix=p.name + \".\", dir=str(p.parent))\n"
                    "    try:\n"
                    "        with os.fdopen(fd, \"w\", encoding=\"utf-8\") as f:\n"
                    "            f.write(data)\n"
                    "            f.flush()\n"
                    "            os.fsync(f.fileno())\n"
                    "        os.replace(tmp, p)\n"
                    "    finally:\n"
                    "        try:\n"
                    "            os.remove(tmp)\n"
                    "        except FileNotFoundError:\n"
                    "            pass\n"
                    "```"
                )
            elif name == "parse_bool":
                user = (
                    "Write Python code with type hints: parse_bool(value: str) -> bool. "
                    "Accept true/false/1/0/yes/no (case-insensitive). Raise ValueError otherwise."
                )
                assistant = (
                    "```python\n"
                    "from __future__ import annotations\n"
                    "\n"
                    "\n"
                    "def parse_bool(value: str) -> bool:\n"
                    "    v = value.strip().lower()\n"
                    "    if v in {\"true\", \"1\", \"yes\", \"y\", \"on\"}:\n"
                    "        return True\n"
                    "    if v in {\"false\", \"0\", \"no\", \"n\", \"off\"}:\n"
                    "        return False\n"
                    "    raise ValueError(f\"Invalid boolean: {value!r}\")\n"
                    "```"
                )
            else:
                user = (
                    "Write Python code with type hints: retry(fn, attempts, base_delay_s). "
                    "Call fn() up to attempts times; on exception, sleep with exponential backoff; re-raise last error."
                )
                assistant = (
                    "```python\n"
                    "from __future__ import annotations\n"
                    "\n"
                    "import time\n"
                    "from typing import Callable, TypeVar\n"
                    "\n"
                    "T = TypeVar(\"T\")\n"
                    "\n"
                    "\n"
                    "def retry(fn: Callable[[], T], attempts: int = 3, base_delay_s: float = 0.25) -> T:\n"
                    "    if attempts <= 0:\n"
                    "        raise ValueError(\"attempts must be > 0\")\n"
                    "    delay = base_delay_s\n"
                    "    last_err: Exception | None = None\n"
                    "    for _ in range(attempts):\n"
                    "        try:\n"
                    "            return fn()\n"
                    "        except Exception as e:\n"
                    "            last_err = e\n"
                    "            time.sleep(delay)\n"
                    "            delay *= 2\n"
                    "    assert last_err is not None\n"
                    "    raise last_err\n"
                    "```"
                )
            out.append(_chatml(system, user, _think_final(assistant)))
        elif k < 0.85:
            name = ident(ts_names)
            if name == "debounce":
                fn = ident(["debounce", "debounceFn", "debounced"])
                user = f"Write minimal TypeScript: {fn}(fn, waitMs). Return a debounced function."
                assistant = (
                    "```ts\n"
                    f"export function {fn}<T extends (...args: any[]) => void>(fn: T, waitMs: number) {{\n"
                    "  let t: ReturnType<typeof setTimeout> | undefined;\n"
                    "  return (...args: Parameters<T>) => {\n"
                    "    if (t) clearTimeout(t);\n"
                    "    t = setTimeout(() => fn(...args), waitMs);\n"
                    "  };\n"
                    "}\n"
                    "```"
                )
            elif name == "clamp":
                fn = ident(["clamp", "clampNumber", "bounded"])
                user = f"Write TypeScript: {fn}(n, min, max) clamps n into [min,max]."
                assistant = (
                    "```ts\n"
                    f"export function {fn}(n: number, min: number, max: number): number {{\n"
                    "  return Math.min(max, Math.max(min, n));\n"
                    "}\n"
                    "```"
                )
            elif name == "pick":
                user = "Write TypeScript: pick(obj, keys) returns a new object with only the provided keys."
                assistant = (
                    "```ts\n"
                    "export function pick<T extends Record<string, any>, K extends keyof T>(obj: T, keys: readonly K[]): Pick<T, K> {\n"
                    "  const out = {} as Pick<T, K>;\n"
                    "  for (const k of keys) out[k] = obj[k];\n"
                    "  return out;\n"
                    "}\n"
                    "```"
                )
            else:
                user = "Write TypeScript: toTitleCase(s) converts a string to Title Case (split on spaces)."
                assistant = (
                    "```ts\n"
                    "export function toTitleCase(s: string): string {\n"
                    "  return s\n"
                    "    .trim()\n"
                    "    .split(/\\s+/)\n"
                    "    .filter(Boolean)\n"
                    "    .map(w => w.slice(0, 1).toUpperCase() + w.slice(1).toLowerCase())\n"
                    "    .join(\" \");\n"
                    "}\n"
                    "```"
                )
            out.append(_chatml(system, user, _think_final(assistant)))
        else:
            users = ident(tbl_names)
            orders = ident([t for t in tbl_names if t != users])
            user = (
                f"Write an SQL query: from tables {users}(id) and {orders}(user_id,total), return top 10 user ids by total spend."
            )
            assistant = (
                "```sql\n"
                f"SELECT u.id, SUM(o.total) AS total_spend\n"
                f"FROM {users} u\n"
                f"JOIN {orders} o ON o.user_id = u.id\n"
                "GROUP BY u.id\n"
                "ORDER BY total_spend DESC\n"
                "LIMIT 10;\n"
                "```"
            )
            out.append(_chatml(system, user, _think_final(assistant)))
    return out


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dedupe_texts(texts: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for t in texts:
        h = _stable_hash(t)
        if h in seen:
            continue
        seen.add(h)
        out.append(t)
    return out


def _validate_jsonl_file(path: Path) -> ValidationResult:
    with path.open("r", encoding="utf-8") as handle:
        return validate_jsonl_rows(handle)


def build_dataset(
    project_root: Path,
    rag_dir: Path,
    seed: int,
    train_size: int,
    valid_size: int,
) -> Tuple[List[str], List[str]]:
    system_prompt, _ = _load_prompts(project_root)
    rng = random.Random(seed)

    rag_docs_path = rag_dir / "docs_metadata.npy"
    if not rag_docs_path.exists():
        raise FileNotFoundError(f"RAG docs not found: {rag_docs_path}")

    rag_docs = np.load(str(rag_docs_path), allow_pickle=True)
    if len(rag_docs) < 1:
        raise ValueError("RAG docs are empty")

    total_target = train_size + valid_size

    code_target = int(total_target * 0.40)
    rag_target = int(total_target * 0.50)
    behavior_target = max(0, total_target - code_target - rag_target)

    texts: List[str] = []
    texts.extend(_seed_texts(system_prompt))
    texts.extend(_anti_repetition_texts(system_prompt))
    texts.extend(_code_texts(system_prompt, rng=rng, count=code_target))

    def add_rag(n_texts: int) -> None:
        if n_texts <= 0:
            return
        chunk_needed = max(1, (n_texts + 1) // 2)
        chunk_needed = min(chunk_needed, len(rag_docs))
        idx = rng.sample(range(len(rag_docs)), k=chunk_needed)
        sampled_chunks = [str(rag_docs[i]) for i in idx]
        texts.extend(_make_rag_texts(system_prompt, sampled_chunks))

    add_rag(rag_target)

    if behavior_target > 0:
        extra = _code_texts(system_prompt, rng=rng, count=behavior_target)
        texts.extend(extra)

    texts = _dedupe_texts(texts)
    rng.shuffle(texts)

    while len(texts) < total_target:
        missing = total_target - len(texts)
        add_rag(min(missing * 2, 20000))
        texts = _dedupe_texts(texts)
        rng.shuffle(texts)

    if len(texts) < 2:
        raise ValueError("Not enough examples after deduplication")

    if len(texts) < train_size + valid_size:
        cap = len(texts)
        train_size = max(1, int(cap * 0.95))
        valid_size = max(1, cap - train_size)

    train = texts[:train_size]
    valid = texts[train_size: train_size + valid_size]
    if not train or not valid:
        raise ValueError("train/valid split produced an empty file")
    return train, valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path("lora_data") / "lora-gem"))
    ap.add_argument("--rag-dir", default=str(Path(os.path.expanduser("~")) / ".lokumf" / "rag"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-size", type=int, default=60000)
    ap.add_argument("--valid-size", type=int, default=3000)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).resolve()
    rag_dir = Path(args.rag_dir).expanduser().resolve()

    train, valid = build_dataset(
        project_root=project_root,
        rag_dir=rag_dir,
        seed=args.seed,
        train_size=args.train_size,
        valid_size=args.valid_size,
    )

    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    write_jsonl_stream(train_path, train)
    write_jsonl_stream(valid_path, valid)

    train_result = _validate_jsonl_file(train_path)
    valid_result = _validate_jsonl_file(valid_path)
    if train_result.invalid or valid_result.invalid:
        raise SystemExit(
            "JSONL validation failed: "
            f"train bad={train_result.invalid}/{train_result.total} "
            f"valid bad={valid_result.invalid}/{valid_result.total}"
        )

    print(str(out_dir))
    print(f"train={train_result.total} valid={valid_result.total}")


if __name__ == "__main__":
    main()
