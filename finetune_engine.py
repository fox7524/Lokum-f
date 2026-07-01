import json
import os
import re
import subprocess
import sys
import shutil
import time
from pathlib import Path
from typing import List

from finetune import ValidationResult, validate_jsonl_rows, write_jsonl_stream
from finetune.job_preflight import preflight_training

def _presplit_text(text: str, max_seq_length: int, batch_size: int) -> list[str]:
    """
    Split overly-long samples into multiple smaller samples to avoid OOM.

    Important quality rule:
    - Never slice raw ChatML strings in the middle of tags.
      (That produces invalid training data and hurts quality.)
    """
    t = text or ""
    max_seq_length = int(max_seq_length) if int(max_seq_length) > 0 else 512
    batch_size = int(batch_size) if int(batch_size) > 0 else 1

    # Rough token->char approximation. Make it configurable.
    chars_per_token = float(os.environ.get("LOKUMAI_FT_PRESPLIT_CHARS_PER_TOKEN", "4.0").strip() or "4.0")
    base_limit = int(max(512, max_seq_length * chars_per_token))

    # If the user tries batch>1, reduce per-sample budget slightly (still never tag-slice).
    eff_limit = int(base_limit * (0.85 if batch_size > 1 else 1.0))
    eff_limit = max(512, eff_limit)

    if len(t) <= eff_limit:
        return [t]

    # ---- ChatML-aware splitting (preferred) ----
    if "<|im_start|>" in t and "<|im_end|>" in t:
        msg_re = re.compile(r"<\|im_start\|>(system|user|assistant)\n([\s\S]*?)<\|im_end\|>\n?")
        msgs = [(m.group(1), m.group(2)) for m in msg_re.finditer(t)]
        if not msgs:
            # Fallback to plain-text splitting below
            pass
        else:
            system_msg = ""
            rest = msgs
            if rest and rest[0][0] == "system":
                system_msg = rest[0][1]
                rest = rest[1:]

            def wrap(role: str, content: str) -> str:
                return f"<|im_start|>{role}\n{content.rstrip()}\n<|im_end|>\n"

            def split_content(content: str, limit: int) -> list[str]:
                s = (content or "").strip()
                if not s:
                    return [""]
                if len(s) <= limit:
                    return [s]
                # Prefer paragraph boundaries.
                parts = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
                out: list[str] = []
                acc = ""
                for p in parts:
                    cand = (acc + ("\n\n" if acc else "") + p) if acc else p
                    if len(cand) <= limit:
                        acc = cand
                        continue
                    if acc:
                        out.append(acc)
                        acc = ""
                    if len(p) <= limit:
                        acc = p
                        continue
                    # Hard split as last resort (still content-only, tags remain intact)
                    k = 0
                    while k < len(p):
                        out.append(p[k : k + limit])
                        k += limit
                if acc:
                    out.append(acc)
                return [x for x in out if x.strip()]

            # If any single message is huge, split its *content* into multiple messages
            # to keep ChatML valid (no tag slicing).
            expanded: list[tuple[str, str]] = []
            per_msg_limit = max(256, int(eff_limit * 0.75))
            for role, content in rest:
                chunks = split_content(content, per_msg_limit)
                for c in chunks:
                    expanded.append((role, c))

            out: list[str] = []
            cur_msgs: list[tuple[str, str]] = []

            def serialize(msgs2: list[tuple[str, str]]) -> str:
                s = ""
                if system_msg:
                    s += wrap("system", system_msg)
                for r, c in msgs2:
                    s += wrap(r, c)
                return s

            for r, c in expanded:
                cand_msgs = cur_msgs + [(r, c)]
                if cur_msgs and len(serialize(cand_msgs)) > eff_limit:
                    out.append(serialize(cur_msgs))
                    cur_msgs = [(r, c)]
                else:
                    cur_msgs = cand_msgs

            if cur_msgs:
                out.append(serialize(cur_msgs))

            return [s for s in out if s.strip()]

    # ---- Plain-text splitting fallback ----
    parts = re.split(r"\n\s*\n", t)
    acc = ""
    out2: list[str] = []
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        cand = (acc + ("\n\n" if acc else "") + p) if acc else p
        if len(cand) <= eff_limit:
            acc = cand
            continue
        if acc:
            out2.append(acc)
            acc = ""
        if len(p) <= eff_limit:
            acc = p
            continue
        k = 0
        while k < len(p):
            out2.append(p[k : k + eff_limit])
            k += eff_limit
    if acc:
        out2.append(acc)
    return [s for s in out2 if s.strip()]

def _presplit_jsonl_file(fp: str, max_seq_length: int, batch_size: int) -> int:
    if not fp or not os.path.isfile(fp):
        return 0
    changed = 0
    out_lines: list[str] = []
    with open(fp, "r", encoding="utf-8") as f:
        for ln in f.read().splitlines():
            s = (ln or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                obj = {"text": s}
            if not isinstance(obj, dict) or "text" not in obj:
                out_lines.append(json.dumps(obj, ensure_ascii=False))
                continue
            text = str(obj.get("text") or "")
            pieces = _presplit_text(text, max_seq_length=max_seq_length, batch_size=batch_size)
            if len(pieces) > 1:
                changed += 1
            for p in pieces:
                obj2 = dict(obj)
                obj2["text"] = p
                out_lines.append(json.dumps(obj2, ensure_ascii=False))
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for ln in out_lines:
            if ln.strip():
                f.write(ln + "\n")
    os.replace(tmp, fp)
    return changed


def _validate_jsonl_file(path: Path) -> ValidationResult:
    with path.open("r", encoding="utf-8") as handle:
        return validate_jsonl_rows(handle)


def _ensure_valid_jsonl_file(path: Path) -> ValidationResult:
    result = _validate_jsonl_file(path)
    if result.invalid:
        raise ValueError(f"Invalid JSONL rows in {path}: {result.invalid}/{result.total}")
    return result


def _perf_log(stage: str, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    print(f"[perf] stage={stage} seconds={elapsed:.3f}")

class FinetuneEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Keep large/private training artifacts out of the git repo by default.
        # Default: ~/.lokumai/lora_data (override via LOKUMAI_LORA_DIR)
        try:
            from lokum_paths import lora_dir as _lora_dir, ensure_dir as _ensure_dir  # type: ignore

            self.dataset_dir = str(_ensure_dir(_lora_dir()))
        except Exception:
            self.dataset_dir = "lora_data"
            os.makedirs(self.dataset_dir, exist_ok=True)
            
    def prepare_dataset(self, text_chunks: List[str]):
        """Converts raw text chunks into a train/valid JSONL dataset for MLX lora."""
        started_at = time.perf_counter()
        train_path = Path(self.dataset_dir) / "train.jsonl"
        valid_path = Path(self.dataset_dir) / "valid.jsonl"
        
        split_idx = int(len(text_chunks) * 0.9)
        train_chunks = text_chunks[:split_idx]
        valid_chunks = text_chunks[split_idx:]
        
        if not train_chunks:
            train_chunks = text_chunks
            valid_chunks = text_chunks[:1]

        train_texts = [
            f"Instruction: Analyze the following knowledge.\n\nKnowledge: {chunk}\n\nResponse: Understood."
            for chunk in train_chunks
        ]
        valid_texts = [
            f"Instruction: Analyze the following knowledge.\n\nKnowledge: {chunk}\n\nResponse: Understood."
            for chunk in valid_chunks
        ]

        write_jsonl_stream(train_path, train_texts)
        write_jsonl_stream(valid_path, valid_texts)
        _ensure_valid_jsonl_file(train_path)
        _ensure_valid_jsonl_file(valid_path)
        _perf_log("finetune_prepare_dataset", started_at)

        return str(train_path), str(valid_path)

    def build_ask_before_acting_dataset(self, qa_pairs: List[dict]):
        """
        Specialized builder for the 50 hand-written 'Ask Before Acting' pairs.
        Each dictionary should have 'user' and 'assistant' keys.
        """
        started_at = time.perf_counter()
        train_path = Path(self.dataset_dir) / "ask_before_acting_train.jsonl"
        texts = [
            "<|im_start|>user\n"
            + pair["user"]
            + "<|im_end|>\n<|im_start|>assistant\n"
            + pair["assistant"]
            + "<|im_end|>\n"
            for pair in qa_pairs
        ]

        write_jsonl_stream(train_path, texts)
        _ensure_valid_jsonl_file(train_path)
        _perf_log("finetune_build_ask_before_acting_dataset", started_at)

        return str(train_path)

    def presplit_dataset(self, dataset_path: str, max_seq_length: int, batch_size: int) -> dict:
        data_dir = os.path.abspath(dataset_path or self.dataset_dir)
        train_fp = os.path.join(data_dir, "train.jsonl")
        valid_fp = os.path.join(data_dir, "valid.jsonl")
        train_changed = _presplit_jsonl_file(train_fp, int(max_seq_length), int(batch_size))
        valid_changed = _presplit_jsonl_file(valid_fp, int(max_seq_length), int(batch_size))
        return {
            "train_changed": int(train_changed),
            "valid_changed": int(valid_changed),
            "train_fp": train_fp,
            "valid_fp": valid_fp,
        }

    def start_training(
        self,
        batch_size=2,
        num_layers=16,
        iters=100,
        dataset_path=None,
        adapter_path=None,
        config_path=None,
        resume_adapter_file: str | None = None,
    ) -> subprocess.Popen:
        """Starts the MLX LoRA training loop as a non-blocking subprocess."""
        data_dir = dataset_path if dataset_path else self.dataset_dir
        started_at = time.perf_counter()
        ok, message = preflight_training(self.model_path, Path(data_dir))
        _perf_log("finetune_preflight_training", started_at)
        if not ok:
            raise RuntimeError(message)
        cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", self.model_path, "--train", "--data", data_dir]
        cmd += ["--batch-size", str(batch_size), "--num-layers", str(num_layers), "--iters", str(iters)]
        if resume_adapter_file:
            cmd += ["--resume-adapter-file", str(resume_adapter_file)]
        if os.environ.get("LOKUMAI_FT_GRAD_CHECKPOINT", "1") != "0":
            cmd += ["--grad-checkpoint"]
        val_batches = os.environ.get("LOKUMAI_FT_VAL_BATCHES", "1").strip()
        if val_batches:
            cmd += ["--val-batches", str(val_batches)]
        steps_per_eval = os.environ.get("LOKUMAI_FT_STEPS_PER_EVAL", "200").strip()
        if steps_per_eval:
            cmd += ["--steps-per-eval", str(steps_per_eval)]
        max_seq = os.environ.get("LOKUMAI_FT_MAX_SEQ_LENGTH", "512").strip()
        if max_seq:
            cmd += ["--max-seq-length", str(max_seq)]
        clear_thr = os.environ.get("LOKUMAI_FT_CLEAR_CACHE_THRESHOLD", "2.0").strip()
        if clear_thr:
            cmd += ["--clear-cache-threshold", str(clear_thr)]
        if adapter_path:
            cmd += ["--adapter-path", str(adapter_path)]
        if config_path:
            cmd += ["--config", str(config_path)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=(sys.platform != "win32"),
        )
        return process

    def start_validation(self, dataset_path: str, adapter_path: str, config_path: str | None = None) -> subprocess.Popen:
        """
        Runs a post-training evaluation pass using the dataset's valid.jsonl as test.jsonl.
        This allows "train first, validate later" without running validation during training.
        """
        data_dir = dataset_path if dataset_path else self.dataset_dir
        valid_fp = os.path.join(os.path.abspath(data_dir), "valid.jsonl")
        if not os.path.isfile(valid_fp):
            raise RuntimeError("valid.jsonl not found in dataset directory.")

        ts = time.strftime("%Y%m%d_%H%M%S")
        # Keep validation artifacts next to other LoRA outputs
        base = os.path.abspath(self.dataset_dir or "lora_data")
        eval_dir = os.path.abspath(os.path.join(base, "validate_only", f"run_{ts}"))
        os.makedirs(eval_dir, exist_ok=True)
        shutil.copyfile(valid_fp, os.path.join(eval_dir, "test.jsonl"))
        try:
            if os.environ.get("LOKUMAI_FT_PRESPLIT", "1") != "0":
                max_seq = int(os.environ.get("LOKUMAI_FT_MAX_SEQ_LENGTH", "512").strip() or "512")
                _presplit_jsonl_file(os.path.join(eval_dir, "test.jsonl"), max_seq, 1)
        except Exception:
            pass

        cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", self.model_path, "--data", eval_dir, "--test"]
        test_batches = os.environ.get("LOKUMAI_FT_TEST_BATCHES", "1").strip()
        if test_batches:
            cmd += ["--test-batches", str(test_batches)]
        max_seq = os.environ.get("LOKUMAI_FT_MAX_SEQ_LENGTH", "512").strip()
        if max_seq:
            cmd += ["--max-seq-length", str(max_seq)]
        clear_thr = os.environ.get("LOKUMAI_FT_CLEAR_CACHE_THRESHOLD", "2.0").strip()
        if clear_thr:
            cmd += ["--clear-cache-threshold", str(clear_thr)]
        if adapter_path:
            cmd += ["--adapter-path", str(adapter_path)]
        if config_path:
            cmd += ["--config", str(config_path)]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=(sys.platform != "win32"),
        )
        return process
