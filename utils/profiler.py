import atexit
import functools
import json
import os
import threading
import time
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


class _Profiler:
    """
    Lightweight profiling utility for timing code spans and function calls.

    Usage:
        from utils.profiler import enable, profile, patch_methods
        enable(output_path="/path/to/timings.jsonl")
        with profile("load_model"):
            ...
        # Or patch methods dynamically (monkey-patch)
        patch_methods([(Cls, "method", "Cls.method")])
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self.records: list[dict] = []
        self.output_path: Optional[str] = None
        self._lock = threading.Lock()
        self._start_ts = time.time()
        self._summary_written = False

        atexit.register(self._on_exit)

    # ------------ public API ------------
    def enable(self, output_path: Optional[str] = None) -> None:
        self.enabled = True
        # Allow env var override
        env_path = os.getenv("MCTS_PROFILE_FILE")
        if env_path:
            output_path = env_path
        if output_path is None:
            output_path = os.path.abspath("timings.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.output_path = output_path

    def disable(self) -> None:
        self.enabled = False

    def profile(self, span: str, meta: Optional[Dict[str, Any]] = None):
        """Context manager for timing a span."""
        if not self.enabled:
            return _NullCtx()
        return _SpanCtx(self, span, meta or {})

    def profile_fn(self, span: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to time a function call."""
        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if not self.enabled:
                return func

            name = span or getattr(func, "__qualname__", getattr(func, "__name__", "<fn>"))

            @functools.wraps(func)
            def _wrapped(*args, **kwargs):
                meta = {}
                # Allow passing profiling meta via reserved kwarg without affecting original signature
                _meta = kwargs.pop("_profile_meta", None)
                if isinstance(_meta, dict):
                    meta.update(_meta)
                with self.profile(name, meta):
                    return func(*args, **kwargs)

            return _wrapped

        return _decorator

    def patch_methods(self, specs: Iterable[Tuple[object, str, Optional[str]]]) -> None:
        """Monkey-patch class/instance methods with profiled wrappers.

        specs: iterable of (obj_or_cls, attr_name, span_name)
        """
        if not self.enabled:
            return
        for obj, attr, span in specs:
            if obj is None or not hasattr(obj, attr):
                continue
            original = getattr(obj, attr)
            # Avoid double-wrapping
            if getattr(original, "__wrapped_by_profiler__", False):
                continue
            wrapped = self.profile_fn(span)(original)
            setattr(wrapped, "__wrapped_by_profiler__", True)
            try:
                setattr(obj, attr, wrapped)
            except Exception:
                # Fallback: skip if cannot set (e.g., read-only builtins)
                continue

    # ------------ internal helpers ------------
    def _record(self, span: str, start: float, end: float, meta: Dict[str, Any]):
        rec = {
            "span": span,
            "t_start": start,
            "t_end": end,
            "dur_ms": round((end - start) * 1000.0, 3),
            "thread": threading.current_thread().name,
            "pid": os.getpid(),
            "rel_s": round(start - self._start_ts, 6),
        }
        if meta:
            rec["meta"] = meta
        with self._lock:
            self.records.append(rec)
            # Write incrementally to reduce data loss in long runs
            if self.output_path:
                try:
                    with open(self.output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    # Ignore file I/O errors during profiling
                    pass

    def _on_exit(self):
        if not self.enabled or self._summary_written:
            return
        try:
            self._write_summary()
        finally:
            self._summary_written = True

    def _write_summary(self):
        if not self.records:
            return
        # Aggregate by span
        agg: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            s = r["span"]
            a = agg.setdefault(s, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
            a["count"] += 1
            a["total_ms"] += r["dur_ms"]
            a["max_ms"] = max(a["max_ms"], r["dur_ms"])
        for s, a in agg.items():
            a["avg_ms"] = round(a["total_ms"] / max(1, a["count"]), 3)
            a["total_ms"] = round(a["total_ms"], 3)
            a["max_ms"] = round(a["max_ms"], 3)

        base = os.path.splitext(self.output_path or "timings.jsonl")[0]
        summary_json = base + ".summary.json"
        summary_csv = base + ".summary.csv"
        try:
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump(agg, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        try:
            # CSV header
            lines = ["span,count,total_ms,avg_ms,max_ms"]
            # Sort by total_ms desc
            for span, a in sorted(agg.items(), key=lambda x: x[1]["total_ms"], reverse=True):
                lines.append(f"{span},{a['count']},{a['total_ms']},{a['avg_ms']},{a['max_ms']}")
            with open(summary_csv, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception:
            pass


class _SpanCtx:
    def __init__(self, profiler: _Profiler, span: str, meta: Dict[str, Any]) -> None:
        self.profiler = profiler
        self.span = span
        self.meta = meta
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        end = time.time()
        self.profiler._record(self.span, self._start, end, self.meta)
        # Do not suppress exceptions
        return False


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


# Singleton-like helpers
_global_profiler = _Profiler()


def enable(output_path: Optional[str] = None) -> None:
    """Enable profiling and set output file path (JSONL)."""
    _global_profiler.enable(output_path)


def disable() -> None:
    _global_profiler.disable()


def profile(span: str, meta: Optional[Dict[str, Any]] = None):
    return _global_profiler.profile(span, meta=meta)


def profile_fn(span: Optional[str] = None):
    return _global_profiler.profile_fn(span)


def patch_methods(specs: Iterable[Tuple[object, str, Optional[str]]]) -> None:
    _global_profiler.patch_methods(specs)
