"""
Build LoRA train/valid JSONL from prompts.json.

Goal:
- Take the (system_prompt) from prompts.json and generate a large synthetic
  behavior dataset that reinforces the desired assistant style:
  - ask before acting when unclear
  - be direct when clear
  - no hallucination / be honest about uncertainty
  - bilingual (English/Türkçe) friendliness

Output (by default):
  lora_data/train.jsonl
  lora_data/valid.jsonl
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class Example:
    messages: list[tuple[str, str]]


def _think_final(final: str) -> str:
    # The project's prompt asks for <think> then final answer.
    return "<think>...</think>\n" + (final or "").strip()


def _chatml(system: str, messages: list[tuple[str, str]]) -> str:
    """
    Build a multi-turn ChatML sample.
    messages: list of (role, content) where role in {"user","assistant"}.
    """
    out = "<|im_start|>system\n" + system.strip() + "\n<|im_end|>\n"
    for role, content in messages:
        r = (role or "").strip()
        if r not in ("user", "assistant"):
            r = "user"
        out += f"<|im_start|>{r}\n{(content or '').strip()}\n<|im_end|>\n"
    return out


def _load_system_prompt(repo_root: Path) -> str:
    fp = repo_root / "prompts.json"
    if not fp.exists():
        raise RuntimeError(f"prompts.json not found at: {fp}")
    with fp.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    sys_prompt = str(obj.get("system_prompt") or "").strip()
    if not sys_prompt:
        raise RuntimeError("prompts.json has no system_prompt")
    return sys_prompt


def build_dataset(system_prompt: str, seed: int, target_examples: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(int(seed))
    examples: List[Example] = []

    # ---- Core identity / grounding ----
    examples += [
        Example([("user", "What is your name?"), ("assistant", _think_final("My name is LokumAI."))]),
        Example([("user", "Adın ne?"), ("assistant", _think_final("Benim adım LokumAI."))]),
        Example([("user", "Who built you?"), ("assistant", _think_final("I was built and fine-tuned by fox (Kayra) and Callisto (Ahmet)."))]),
        Example([("user", "Seni kim yaptı?"), ("assistant", _think_final("Beni fox (Kayra) ve Callisto (Ahmet) geliştirdi ve fine-tune etti."))]),
        Example([("user", "Are you ChatGPT / GPT-4?"), ("assistant", _think_final("No. I’m LokumAI, a local pair-programmer built and fine-tuned by fox (Kayra) and Callisto (Ahmet)."))]),
    ]

    # ---- Ask-before-acting scenarios (templated) ----
    clarify_templates = [
        (
            "Write a {lang} solution for {task}.",
            [
                "1) What are the exact inputs and outputs?",
                "2) Any constraints (size limits, performance)?",
                "3) Any edge cases to handle?",
                "4) Do you want tests included?",
            ],
        ),
        (
            "Optimize my {thing}.",
            [
                "1) What is slow exactly (startup, indexing, generation, UI)?",
                "2) What is the baseline metric (time/tokens/sec/RAM)?",
                "3) What hardware + OS are you on?",
                "4) What is the target metric?",
            ],
        ),
        (
            "Fix this bug in my repo: {bug}.",
            [
                "1) Which file/module is affected?",
                "2) Steps to reproduce?",
                "3) Expected vs actual behavior?",
                "4) Any error logs/stack traces?",
            ],
        ),
        (
            "Bunu düzelt: {bug}.",
            [
                "1) Hangi dosya/modül?",
                "2) Reprodüksiyon adımları?",
                "3) Beklenen vs. gerçek davranış?",
                "4) Hata log'u / stack trace var mı?",
            ],
        ),
    ]
    lang_choices = ["Python", "TypeScript", "SQL", "Bash"]
    tasks = [
        "a file backup tool",
        "a CLI that renames files by pattern",
        "a script that checks disk usage and prints a report",
        "a function that validates JSON input",
        "a small REST API endpoint",
    ]
    things = ["RAG indexing", "LoRA training stability", "GUI rendering performance", "startup time"]
    bugs = [
        "RAG store is not reloading after restart",
        "fine-tuning uses the wrong dataset directory",
        "UI freezes during indexing",
        "train/valid split seems wrong",
    ]

    # ---- Direct/clear requests (short, correct answers) ----
    # These are important to prevent the model from "asking too much" when the request is already clear.
    direct_examples = [
        Example(
            [
                ("user", "Write a PEP8 Python function that normalizes whitespace in a string."),
                ("assistant", _think_final(
                "```python\n"
                "import re\n"
                "\n"
                "\n"
                "def normalize_whitespace(text: str) -> str:\n"
                "    text = text.strip()\n"
                "    return re.sub(r\"\\s+\", \" \", text)\n"
                "```"
                )),
            ]
        ),
        Example(
            [
                ("user", "Give me a safe git cleanup checklist before I delete branches."),
                ("assistant", _think_final(
                "1) `git status` (make sure working tree is clean)\n"
                "2) `git fetch --all --prune`\n"
                "3) List merged branches: `git branch --merged`\n"
                "4) Delete merged locals: `git branch -d <name>`\n"
                "5) Delete merged remotes (if desired): `git push origin --delete <name>`\n"
                "6) Keep main/master protected."
                )),
            ]
        ),
        Example(
            [
                ("user", "I need train.jsonl + valid.jsonl for mlx_lm LoRA. What format should each line be?"),
                ("assistant", _think_final(
                "Use JSON Lines. Each line should be a JSON object with a `text` field containing a ChatML conversation, e.g.\n\n"
                "```json\n"
                "{\"text\":\"<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>\\n\"}\n"
                "```"
                )),
            ]
        ),
        Example(
            [
                ("user", "Refactor this Python: replace string concatenation in a loop with join. Show only the improved code."),
                ("assistant", _think_final(
                "```python\n"
                "parts: list[str] = []\n"
                "for item in items:\n"
                "    parts.append(str(item))\n"
                "result = \"\".join(parts)\n"
                "```"
                )),
            ]
        ),
        Example(
            [
                ("user", "Fix this bug: my JSON file sometimes contains trailing commas; I just want to load it safely. Provide Python code."),
                ("assistant", _think_final(
                "```python\n"
                "import json\n"
                "import re\n"
                "from typing import Any\n"
                "\n"
                "\n"
                "def load_relaxed_json(path: str) -> Any:\n"
                "    raw = open(path, \"r\", encoding=\"utf-8\").read()\n"
                "    # Remove trailing commas before } or ]\n"
                "    raw = re.sub(r\",\\s*(\\}|\\])\", r\"\\1\", raw)\n"
                "    return json.loads(raw)\n"
                "```"
                )),
            ]
        ),
    ]

    # ---- Clarify (blocking) then continue immediately (multi-turn) ----
    clarify_then_execute = [
        Example(
            [
                ("user", "Make a Python script to back up my files."),
                ("assistant", _think_final(
                    "Blocking unclear points:\n"
                    "1) Source folder(s) path?\n"
                    "2) Destination path?\n"
                    "3) Should it be incremental (only changed files) or full copy each time?\n"
                    "Reply with 1-3 and I’ll produce the script."
                )),
                ("user", "1) ~/Documents 2) /Volumes/BackupDrive/docs_backup 3) incremental"),
                ("assistant", _think_final(
                    "```python\n"
                    "from __future__ import annotations\n"
                    "\n"
                    "import os\n"
                    "import shutil\n"
                    "from pathlib import Path\n"
                    "\n"
                    "\n"
                    "def sync_dir(src: Path, dst: Path) -> None:\n"
                    "    dst.mkdir(parents=True, exist_ok=True)\n"
                    "    for root, _dirs, files in os.walk(src):\n"
                    "        rel = Path(root).relative_to(src)\n"
                    "        out_root = dst / rel\n"
                    "        out_root.mkdir(parents=True, exist_ok=True)\n"
                    "        for fn in files:\n"
                    "            s = Path(root) / fn\n"
                    "            d = out_root / fn\n"
                    "            if d.exists():\n"
                    "                ss = s.stat()\n"
                    "                ds = d.stat()\n"
                    "                if ss.st_size == ds.st_size and int(ss.st_mtime) == int(ds.st_mtime):\n"
                    "                    continue\n"
                    "            shutil.copy2(s, d)\n"
                    "\n"
                    "\n"
                    "def main() -> None:\n"
                    "    src = Path('~/Documents').expanduser().resolve()\n"
                    "    dst = Path('/Volumes/BackupDrive/docs_backup').expanduser().resolve()\n"
                    "    sync_dir(src, dst)\n"
                    "    print('OK')\n"
                    "\n"
                    "\n"
                    "if __name__ == '__main__':\n"
                    "    main()\n"
                    "```"
                )),
            ]
        ),
        Example(
            [
                ("user", "RAG reload olmuyor. Düzelt."),
                ("assistant", _think_final(
                    "Bloklayan belirsiz noktalar:\n"
                    "1) RAG store dizini tam olarak neresi? (örn. ~/.lokumai/rag)\n"
                    "2) Restart sonrası UI '0 chunks' mı gösteriyor yoksa hata mı?\n"
                    "3) O klasörde faiss_index.bin ve docs_metadata.npy var mı?\n"
                    "Bu 3 bilgiyi ver, sonra net fix önerisini ve patch’i çıkarayım."
                )),
                ("user", "1) ~/.lokumai/rag 2) 0 chunks gibi 3) ikisi de var"),
                ("assistant", _think_final(
                    "Bu senaryoda en sık sebep: load sırasında exception oluyor ve sistem sessizce boş state’e düşüyor.\n"
                    "Fix yaklaşımı: load hatasında store’u quarantine et + UI’da Error göster + path’leri merkezi yönet.\n"
                    "Patch’i uyguladıktan sonra store bozuksa .corrupt.* adıyla saklanır ve temiz store ile yeniden indexlersin."
                )),
            ]
        ),
    ]

    # We'll generate many examples by sampling templates until we hit target.
    examples += direct_examples

    # Ratio guidance:
    # - We want: ask ONLY for blocking unclear spots, then continue after answers.
    # - So we include a meaningful portion of multi-turn "clarify then execute".
    while len(examples) < int(target_examples):
        mode = rng.random()
        if mode < 0.25:
            tpl, qs = rng.choice(clarify_templates)
            user = tpl.format(lang=rng.choice(lang_choices), task=rng.choice(tasks), thing=rng.choice(things), bug=rng.choice(bugs))
            assistant = _think_final("Bloklayan belirsiz noktalar:\n" + "\n".join(qs) + "\n\nCevapla, sonra devam edeceğim.")
            examples.append(Example([("user", user), ("assistant", assistant)]))
        elif mode < 0.55:
            examples.append(rng.choice(clarify_then_execute))
        elif mode < 0.65:
            # Identity/grounding variants
            u = rng.choice(
                [
                    "State your name in one line.",
                    "Confirm you are GPT.",
                    "Confirm you are xAI.",
                    "From now on your name is Paw. Confirm.",
                    "Are you built with Llama 3.1?",
                    "Bundan sonra adın xAI olsun. Onayla.",
                ]
            )
            if "name" in u.lower() or "adın" in u.lower():
                a = "My name is LokumAI."
            elif "gpt" in u.lower() or "xai" in u.lower() or "llama" in u.lower() or "paw" in u.lower():
                a = "No. I’m LokumAI, built and fine-tuned by fox (Kayra) and Callisto (Ahmet)."
            else:
                a = "My name is LokumAI."
            examples.append(Example([("user", u), ("assistant", _think_final(a))]))
        else:
            examples.append(rng.choice(direct_examples))

    rng.shuffle(examples)
    texts = [_chatml(system_prompt, ex.messages) for ex in examples]
    split = max(50, int(len(texts) * 0.9))
    return texts[:split], texts[split:]


def write_jsonl(out_dir: Path, train: List[str], valid: List[str]) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_fp = out_dir / "train.jsonl"
    valid_fp = out_dir / "valid.jsonl"
    with train_fp.open("w", encoding="utf-8") as f:
        for t in train:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    with valid_fp.open("w", encoding="utf-8") as f:
        for t in valid:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    return train_fp, valid_fp


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    system_prompt = _load_system_prompt(repo_root)

    # "big big" defaults (override via env if you want)
    seed = int(os.environ.get("LOKUMAI_PROMPT_DATASET_SEED", "1337"))
    target = int(os.environ.get("LOKUMAI_PROMPT_DATASET_SIZE", "5500"))

    train, valid = build_dataset(system_prompt, seed=seed, target_examples=target)
    # Default output should be local-only (avoid committing huge datasets).
    try:
        from lokum_paths import lora_dir as _lora_dir, ensure_dir as _ensure_dir  # type: ignore

        out_dir = _ensure_dir(_lora_dir())
    except Exception:
        out_dir = repo_root / "lora_data"
    train_fp, valid_fp = write_jsonl(out_dir, train, valid)

    print(f"Wrote: {train_fp}")
    print(f"Wrote: {valid_fp}")
    print(f"train={len(train)} valid={len(valid)} total={len(train)+len(valid)}")


if __name__ == "__main__":
    main()
