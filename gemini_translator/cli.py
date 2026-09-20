from __future__ import annotations

import argparse
import contextlib
import datetime
import fnmatch
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CliError(Exception):
    def __init__(self, message: str, *, exit_code: int = 2, payload: dict | None = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.payload = payload or {}


@dataclass
class TaskPlan:
    chapters: list[str]
    payloads: list[tuple]
    task_chains: list[list[tuple]]
    settings: dict
    summary: dict


def _abs_path(path: str | os.PathLike | None) -> str:
    if not path:
        return ""
    return str(Path(path).expanduser().resolve())


def _load_json_file(path: str | os.PathLike | None, default: Any = None) -> Any:
    if not path:
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise CliError(f"JSON file not found: {path}")
    except json.JSONDecodeError as exc:
        raise CliError(f"Invalid JSON in {path}: {exc}")


def _load_text_file(path: str | os.PathLike | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except FileNotFoundError:
        raise CliError(f"Text file not found: {path}")


def _write_json(payload: dict, *, pretty: bool = True) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
    try:
        sys.stdout.write(text + "\n")
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()


@contextlib.contextmanager
def _redirect_internal_stdout_to_stderr():
    """Keep stdout machine-readable while existing app code keeps using print()."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _ensure_api_config_initialized():
    from .api import config as api_config

    api_config.initialize_configs()
    return api_config


def get_epub_chapters(epub_path: str) -> list[str]:
    from .utils.epub_tools import get_epub_chapter_order

    epub_path = _abs_path(epub_path)
    if not os.path.exists(epub_path):
        raise CliError(f"EPUB file not found: {epub_path}")
    chapters = get_epub_chapter_order(epub_path)
    if not chapters:
        raise CliError(f"No translatable HTML/XHTML chapters found in: {epub_path}")
    return list(chapters)


def _matches_patterns(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    normalized = path.replace("\\", "/")
    basename = os.path.basename(normalized)
    for pattern in patterns:
        pattern = str(pattern or "").strip()
        if not pattern:
            continue
        if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(basename, pattern):
            return True
        if pattern.lower() in normalized.lower():
            return True
    return False


def select_chapters(
    epub_path: str,
    project_manager=None,
    *,
    mode: str = "pending",
    patterns: list[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> list[str]:
    mode = (mode or "pending").strip().lower()
    if mode not in {"all", "pending", "translated"}:
        raise CliError(f"Unsupported chapter mode: {mode}")

    chapters = get_epub_chapters(epub_path)
    patterns = patterns or []
    selected = [chapter for chapter in chapters if _matches_patterns(chapter, patterns)]

    if project_manager is not None and mode != "all":
        filtered = []
        for chapter in selected:
            has_translation = bool(project_manager.get_versions_for_original(chapter))
            if mode == "pending" and not has_translation:
                filtered.append(chapter)
            elif mode == "translated" and has_translation:
                filtered.append(chapter)
        selected = filtered

    offset = max(0, int(offset or 0))
    if offset:
        selected = selected[offset:]
    if limit is not None:
        limit = max(0, int(limit))
        selected = selected[:limit]
    return selected


def load_project_glossary(project_folder: str, glossary_path: str | None = None) -> dict[str, dict]:
    path = glossary_path or os.path.join(project_folder, "project_glossary.json")
    if not path or not os.path.exists(path):
        return {}
    data = _load_json_file(path, default=[])
    if isinstance(data, dict):
        result = {}
        for original, entry in data.items():
            if isinstance(entry, dict):
                result[str(original)] = {
                    "rus": str(entry.get("rus") or entry.get("translation") or ""),
                    "note": str(entry.get("note") or ""),
                }
            else:
                result[str(original)] = {"rus": str(entry or ""), "note": ""}
        return result
    if isinstance(data, list):
        result = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            original = str(entry.get("original") or "").strip()
            if not original:
                continue
            result[original] = {
                "rus": str(entry.get("rus") or entry.get("translation") or ""),
                "note": str(entry.get("note") or ""),
            }
        return result
    raise CliError(f"Unsupported glossary format: {path}")


def _build_sequential_chapter_chains(chapters: list[str], split_count: int) -> list[list[str]]:
    if not chapters:
        return []
    try:
        split_count = int(split_count)
    except (TypeError, ValueError):
        split_count = 1
    split_count = max(1, min(split_count, len(chapters)))
    chains = []
    for index in range(split_count):
        start = (index * len(chapters)) // split_count
        end = ((index + 1) * len(chapters)) // split_count
        if start < end:
            chains.append(chapters[start:end])
    return chains


def _chapter_sizes(epub_path: str, chapters: list[str], project_manager=None) -> dict[str, int]:
    from .utils.epub_tools import get_epub_chapter_sizes_with_cache

    sizes = {}
    if project_manager is not None:
        sizes.update(get_epub_chapter_sizes_with_cache(project_manager, epub_path) or {})

    missing = [chapter for chapter in chapters if int(sizes.get(chapter, 0) or 0) <= 0]
    if missing:
        with zipfile.ZipFile(epub_path, "r") as archive:
            for chapter in missing:
                try:
                    sizes[chapter] = len(archive.read(chapter).decode("utf-8", "ignore"))
                except Exception:
                    sizes[chapter] = 0
    return {chapter: int(sizes.get(chapter, 0) or 0) for chapter in chapters}


def _payload_chapters(payload: tuple) -> list[str]:
    task_type = payload[0] if payload else ""
    if task_type in {"epub", "epub_chunk"}:
        return [str(payload[2])]
    if task_type == "epub_batch":
        return [str(item) for item in payload[2]]
    return []


def summarize_payloads(payloads: list[tuple]) -> dict:
    type_counts = Counter(payload[0] for payload in payloads if payload)
    unique_chapters = []
    seen = set()
    for payload in payloads:
        for chapter in _payload_chapters(payload):
            if chapter not in seen:
                seen.add(chapter)
                unique_chapters.append(chapter)
    return {
        "task_count": len(payloads),
        "chapter_count": len(unique_chapters),
        "task_types": dict(sorted(type_counts.items())),
        "chapters": unique_chapters,
    }


def _apply_mode_overrides(settings: dict, mode: str | None, args) -> None:
    mode = (mode or "saved").strip().lower()
    if mode == "saved":
        return
    if mode == "single":
        settings.update({
            "use_batching": False,
            "chunking": False,
            "sequential_translation": False,
        })
    elif mode == "batch":
        settings.update({
            "use_batching": True,
            "chunking": False,
            "sequential_translation": False,
        })
    elif mode == "chunk":
        settings.update({
            "use_batching": False,
            "chunking": True,
            "sequential_translation": False,
        })
    elif mode == "sequential":
        settings.update({
            "use_batching": False,
            "chunking": False,
            "sequential_translation": True,
            "sequential_translation_splits": int(getattr(args, "splits", 1) or 1),
        })
    else:
        raise CliError(f"Unsupported task mode: {mode}")


def _provider_models(api_config, provider_id: str) -> dict:
    api_config.ensure_dynamic_provider_models(provider_id)
    provider = api_config.api_providers().get(provider_id, {})
    models = provider.get("models", {})
    return models if isinstance(models, dict) else {}


def _resolve_provider(api_config, saved_settings: dict, provider_arg: str | None) -> str:
    providers = api_config.api_providers()
    provider_id = str(provider_arg or saved_settings.get("provider") or "").strip()
    if provider_id and provider_id in providers:
        return provider_id
    if provider_id:
        for candidate_id, provider in providers.items():
            if str(provider.get("display_name") or "").strip() == provider_id:
                return candidate_id
        raise CliError(f"Unknown provider: {provider_id}", payload={"available_providers": sorted(providers)})
    if "gemini" in providers:
        return "gemini"
    visible = [pid for pid, cfg in providers.items() if cfg.get("visible", True)]
    return visible[0] if visible else next(iter(providers))


def _resolve_model(api_config, provider_id: str, saved_settings: dict, model_arg: str | None) -> tuple[str, dict]:
    models_for_provider = _provider_models(api_config, provider_id)
    all_models = api_config.all_models()
    explicit = str(model_arg or "").strip()
    saved = str(saved_settings.get("model") or "").strip()

    for model_name in [explicit, saved, api_config.default_model_name()]:
        if not model_name:
            continue
        model_config = all_models.get(model_name)
        if model_config and str(model_config.get("provider") or provider_id) == provider_id:
            return model_name, model_config
        if model_name in models_for_provider:
            cfg = dict(models_for_provider[model_name] or {})
            cfg.setdefault("provider", provider_id)
            return model_name, cfg

    if models_for_provider:
        model_name, model_config = next(iter(models_for_provider.items()))
        cfg = dict(model_config or {})
        cfg.setdefault("provider", provider_id)
        return model_name, cfg

    raise CliError(f"Provider has no models: {provider_id}")


def _read_api_key_file(path: str | None) -> list[str]:
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    except FileNotFoundError:
        raise CliError(f"API key file not found: {path}")


def _configured_provider_keys(settings_manager, provider_id: str) -> list[str]:
    key_statuses = settings_manager.load_key_statuses()
    keys = [
        item.get("key")
        for item in key_statuses
        if isinstance(item, dict) and item.get("provider") == provider_id and item.get("key")
    ]
    return list(dict.fromkeys(keys))


def _resolve_api_keys(api_config, settings_manager, provider_id: str, saved_settings: dict, args) -> list[str]:
    if not api_config.provider_requires_api_key(provider_id):
        placeholder = api_config.provider_placeholder_api_key(provider_id)
        return [placeholder] if placeholder else []

    keys = []
    keys.extend(getattr(args, "api_key", None) or [])
    keys.extend(_read_api_key_file(getattr(args, "api_key_file", None)))
    keys = [str(key).strip() for key in keys if str(key).strip()]
    if keys:
        return list(dict.fromkeys(keys))

    active_by_provider = saved_settings.get("active_keys_by_provider")
    if isinstance(active_by_provider, dict):
        active = active_by_provider.get(provider_id)
        if isinstance(active, (list, tuple, set)):
            keys = [str(key).strip() for key in active if str(key).strip()]

    if getattr(args, "all_keys", False) or not keys:
        keys = _configured_provider_keys(settings_manager, provider_id)

    if not keys:
        raise CliError(
            f"No API keys available for provider: {provider_id}",
            payload={"hint": "Pass --api-key/--api-key-file or configure keys in the app."},
        )
    return list(dict.fromkeys(keys))


def build_session_settings(settings_manager, project_manager, chapters: list[str], args) -> dict:
    api_config = _ensure_api_config_initialized()
    saved_settings = settings_manager.load_full_session_settings() or {}
    if not isinstance(saved_settings, dict):
        saved_settings = {}

    provider_id = _resolve_provider(api_config, saved_settings, getattr(args, "provider", None))
    model_name, model_config = _resolve_model(api_config, provider_id, saved_settings, getattr(args, "model", None))
    api_keys = _resolve_api_keys(api_config, settings_manager, provider_id, saved_settings, args)

    prompt_text = _load_text_file(getattr(args, "prompt_file", None))
    if prompt_text is None:
        prompt_text = saved_settings.get("custom_prompt") or settings_manager.get_custom_prompt() or api_config.default_prompt()

    project_folder = _abs_path(getattr(args, "project", None))
    epub_path = _abs_path(getattr(args, "epub", None))
    glossary = load_project_glossary(project_folder, getattr(args, "glossary", None))

    settings = dict(saved_settings)
    settings.update({
        "provider": provider_id,
        "model": model_name,
        "model_config": model_config,
        "file_path": epub_path,
        "output_folder": project_folder,
        "api_keys": api_keys,
        "full_glossary_data": glossary,
        "custom_prompt": prompt_text,
        "auto_translation": {"enabled": False},
        "auto_start": True,
        "project_manager": project_manager,
        "use_batching": bool(saved_settings.get("use_batching", False)),
        "chunking": bool(saved_settings.get("chunking", False)),
        "chunk_on_error": bool(saved_settings.get("chunk_on_error", False)),
        "sequential_translation": bool(saved_settings.get("sequential_translation", False)),
        "sequential_translation_splits": int(saved_settings.get("sequential_translation_splits", 1) or 1),
        "task_size_limit": int(saved_settings.get("task_size_limit", 30000) or 30000),
        "dynamic_glossary": bool(saved_settings.get("dynamic_glossary", True)),
        "use_jieba": bool(saved_settings.get("use_jieba", False)),
        "segment_cjk_text": bool(saved_settings.get("segment_cjk_text", False)),
        "fuzzy_threshold": int(saved_settings.get("fuzzy_threshold", 100) or 100),
        "rpm_limit": int(getattr(args, "rpm", None) or saved_settings.get("rpm_limit", model_config.get("rpm", 10)) or 10),
        "rpd_limit": int(saved_settings.get("rpd_limit", 0) or 0),
        "temperature": float(getattr(args, "temperature", None) if getattr(args, "temperature", None) is not None else saved_settings.get("temperature", 1.0)),
        "temperature_override_enabled": bool(getattr(args, "temperature", None) is not None or saved_settings.get("temperature_override_enabled", False)),
        "use_system_instruction": bool(saved_settings.get("use_system_instruction", False)),
        "system_instruction": saved_settings.get("system_instruction"),
        "thinking_enabled": bool(saved_settings.get("thinking_enabled", False)),
        "thinking_budget": saved_settings.get("thinking_budget"),
        "thinking_level": saved_settings.get("thinking_level"),
        "force_accept": bool(getattr(args, "force_accept", False) or saved_settings.get("force_accept", False)),
        "use_json_epub_pipeline": bool(getattr(args, "json_epub", False) or saved_settings.get("use_json_epub_pipeline", False)),
        "use_prettify": bool(saved_settings.get("use_prettify", False)),
        "max_concurrent_requests": saved_settings.get("max_concurrent_requests"),
        "model_id": model_config.get("id"),
    })

    _apply_mode_overrides(settings, getattr(args, "mode", None), args)
    if getattr(args, "task_size", None):
        settings["task_size_limit"] = int(args.task_size)
        settings["task_size_limit_user_defined"] = True

    provider_limit = api_config.provider_max_instances(provider_id)
    requested_workers = getattr(args, "workers", None) or saved_settings.get("num_instances") or len(api_keys)
    try:
        requested_workers = int(requested_workers)
    except (TypeError, ValueError):
        requested_workers = 1
    max_workers = len(api_keys)
    if provider_limit:
        max_workers = min(max_workers, provider_limit)
    settings["num_instances"] = max(1, min(max_workers, requested_workers))

    if settings.get("sequential_translation"):
        settings["sequential_chapter_order"] = get_epub_chapters(epub_path)
        chapter_chains = _build_sequential_chapter_chains(
            chapters,
            settings.get("sequential_translation_splits", 1),
        )
        if chapter_chains:
            settings["sequential_translation_splits"] = len(chapter_chains)
        settings["sequential_chain_starts"] = [chain[0] for chain in chapter_chains if chain]

    settings_override = _load_json_file(getattr(args, "settings_json", None), default=None)
    if isinstance(settings_override, dict):
        settings.update(settings_override)
        if "model_config" not in settings_override:
            settings["model_config"] = api_config.all_models().get(settings.get("model"), settings["model_config"])
        settings["model_id"] = (settings.get("model_config") or {}).get("id")

    return settings


def build_task_plan(epub_path: str, chapters: list[str], settings: dict, project_manager=None) -> TaskPlan:
    from .utils.glossary_tools import TaskPreparer

    sizes = _chapter_sizes(epub_path, chapters, project_manager)
    preparer = TaskPreparer(settings, sizes)
    if settings.get("sequential_translation"):
        chapter_chains = _build_sequential_chapter_chains(
            chapters,
            settings.get("sequential_translation_splits", 1),
        )
        task_chains = [preparer.prepare_tasks(chain) for chain in chapter_chains if chain]
        payloads = [payload for chain in task_chains for payload in chain]
    else:
        task_chains = []
        payloads = preparer.prepare_tasks(chapters)

    summary = summarize_payloads(payloads)
    summary.update({
        "sequential": bool(settings.get("sequential_translation")),
        "chain_count": len(task_chains),
        "total_source_chars": sum(sizes.values()),
    })
    return TaskPlan(chapters=chapters, payloads=payloads, task_chains=task_chains, settings=settings, summary=summary)


def _safe_settings_for_output(settings: dict) -> dict:
    hidden = dict(settings)
    if hidden.get("api_keys"):
        hidden["api_keys"] = [f"...{str(key)[-4:]}" for key in hidden.get("api_keys", [])]
    hidden.pop("project_manager", None)
    if hidden.get("custom_prompt"):
        hidden["custom_prompt_chars"] = len(str(hidden.pop("custom_prompt")))
    if hidden.get("full_glossary_data"):
        hidden["glossary_terms"] = len(hidden.pop("full_glossary_data"))
    return hidden


def _queue_status_counts(task_manager) -> dict[str, int]:
    counts = {"pending": 0, "in_progress": 0, "failed": 0, "completed": 0, "held": 0}
    try:
        with task_manager._get_read_only_conn() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS count FROM tasks GROUP BY status").fetchall()
        for row in rows:
            counts[str(row["status"])] = int(row["count"])
    except Exception:
        pass
    return counts


class CliSessionObserver:
    def __init__(self, app, *, verbose: bool = False, timeout_sec: int | None = None):
        from PyQt6 import QtCore

        self.app = app
        self.QtCore = QtCore
        self.verbose = verbose
        self.timeout_sec = timeout_sec
        self.started_at = time.time()
        self.session_id = None
        self.finished = False
        self.timed_out = False
        self.reason = None
        self.task_events = []
        self.event_counts = Counter()
        self.logs = []
        app.event_bus.event_posted.connect(self.on_event)
        if timeout_sec:
            QtCore.QTimer.singleShot(int(timeout_sec * 1000), self.on_timeout)

    def on_event(self, event: dict):
        event_name = event.get("event")
        self.event_counts[event_name] += 1
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}

        if event_name == "log_message":
            message = str(data.get("message") or "")
            if message:
                self.logs.append(message)
                if self.verbose:
                    print(message, file=sys.stderr)
            return

        if event_name == "session_started":
            self.session_id = data.get("session_id") or event.get("session_id")
            return

        if event_name == "task_finished":
            task_info = data.get("task_info")
            task_payload = None
            if isinstance(task_info, (tuple, list)) and len(task_info) >= 2:
                task_payload = task_info[1]
            self.task_events.append({
                "success": bool(data.get("success")),
                "error_type": data.get("error_type"),
                "message": data.get("message"),
                "task_type": task_payload[0] if task_payload else None,
                "chapters": _payload_chapters(tuple(task_payload)) if task_payload else [],
            })
            return

        if event_name == "session_finished":
            self.finished = True
            self.reason = data.get("reason") or "session_finished"
            self.QtCore.QTimer.singleShot(750, self.app.quit)

    def on_timeout(self):
        if self.finished:
            return
        self.timed_out = True
        self.reason = f"timeout after {self.timeout_sec}s"
        try:
            self.app.event_bus.event_posted.emit({
                "event": "manual_stop_requested",
                "source": "cli",
                "data": {"reason": self.reason},
            })
        finally:
            self.QtCore.QTimer.singleShot(5000, self.app.quit)

    def result_payload(self, task_manager) -> dict:
        counts = _queue_status_counts(task_manager)
        return {
            "finished": self.finished,
            "timed_out": self.timed_out,
            "reason": self.reason,
            "session_id": self.session_id,
            "elapsed_sec": round(time.time() - self.started_at, 3),
            "queue": counts,
            "task_events": {
                "total": len(self.task_events),
                "success": sum(1 for event in self.task_events if event.get("success")),
                "failed": sum(1 for event in self.task_events if not event.get("success")),
            },
            "event_counts": dict(sorted(self.event_counts.items())),
            "recent_logs": self.logs[-20:],
        }


class HeadlessRuntime:
    def __init__(self):
        self.app = None
        self.app_main = None

    def bootstrap(self, *, include_engine: bool):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import main as app_main

        self.app_main = app_main
        app_main.prepare_console_streams()
        app_main.os_patch.PatientLock.register_vip_thread(threading.get_ident())

        app = app_main.ApplicationWithContext(["translator-cli"])
        self.app = app

        if sys.platform == "win32":
            app_main.asyncio.set_event_loop_policy(app_main.asyncio.WindowsSelectorEventLoopPolicy())

        app_main.os_patch.apply()
        app_main.api_config.initialize_configs()
        app.event_bus = app_main.EventBus()
        app.initialize_managers()
        app.settings_manager = app.get_settings_manager()
        app.global_version = getattr(app_main, "APP_VERSION", "")

        if include_engine:
            app_main.initialize_global_resources(app)
            app.task_manager = app_main.ChapterQueueManager(event_bus=app.event_bus)
            app.proxy_controller = app_main.GlobalProxyController(app.event_bus)
            app.settings_manager.load_proxy_settings()
            temp_folder = os.path.join(os.path.expanduser("~"), ".epub_translator_temp")
            os.makedirs(temp_folder, exist_ok=True)
            app.context_manager = app_main.ContextManager(temp_folder)
            app.server_manager = app_main.ServerManager(app.event_bus)
            app.engine = app_main.TranslationEngine(task_manager=app.task_manager)
            app.engine_thread = app_main.QtCore.QThread(app)
            app.engine.moveToThread(app.engine_thread)
            app.engine_thread.finished.connect(app.engine.deleteLater)
            app.engine_thread.start()
            app_main.QtCore.QMetaObject.invokeMethod(
                app.engine,
                "log_thread_identity",
                app_main.QtCore.Qt.ConnectionType.QueuedConnection,
            )

        return app

    def shutdown(self):
        if not self.app:
            return
        app = self.app
        app_main = self.app_main
        try:
            if hasattr(app, "settings_manager") and app.settings_manager:
                app.settings_manager.flush()
        except Exception:
            pass
        if app_main and hasattr(app, "engine_thread") and app.engine_thread.isRunning():
            try:
                app_main.QtCore.QMetaObject.invokeMethod(
                    app.engine,
                    "cleanup",
                    app_main.QtCore.Qt.ConnectionType.BlockingQueuedConnection,
                )
            except Exception:
                pass
            app.engine_thread.quit()
            app.engine_thread.wait(10000)


def _project_manager(project_folder: str):
    from .utils.project_manager import TranslationProjectManager

    project_folder = _abs_path(project_folder)
    os.makedirs(project_folder, exist_ok=True)
    return TranslationProjectManager(project_folder)


def command_status(args) -> dict:
    runtime = HeadlessRuntime()
    app = runtime.bootstrap(include_engine=False)
    api_config = _ensure_api_config_initialized()
    settings = app.settings_manager.load_full_session_settings() or {}
    providers = {}
    key_statuses = app.settings_manager.load_key_statuses()
    keys_by_provider = Counter(
        item.get("provider")
        for item in key_statuses
        if isinstance(item, dict) and item.get("provider")
    )
    for provider_id, provider in api_config.api_providers().items():
        models = provider.get("models", {})
        providers[provider_id] = {
            "display_name": provider.get("display_name") or provider_id,
            "visible": bool(provider.get("visible", True)),
            "requires_api_key": api_config.provider_requires_api_key(provider_id),
            "configured_keys": int(keys_by_provider.get(provider_id, 0)),
            "models": list(models.keys()) if isinstance(models, dict) else [],
            "file_suffix": provider.get("file_suffix"),
        }

    payload = {
        "ok": True,
        "settings_file": app.settings_manager.config_file,
        "saved_provider": settings.get("provider"),
        "saved_model": settings.get("model"),
        "providers": providers,
        "project_history": app.settings_manager.load_project_history(),
    }

    if getattr(args, "project", None):
        pm = _project_manager(args.project)
        project_map = pm.get_full_map()
        payload["project"] = {
            "path": _abs_path(args.project),
            "mapped_chapters": len(project_map),
            "glossary_terms": len(load_project_glossary(_abs_path(args.project))),
        }
    if getattr(args, "epub", None):
        payload["epub"] = {
            "path": _abs_path(args.epub),
            "chapters": len(get_epub_chapters(args.epub)),
        }
    runtime.shutdown()
    return payload


def command_plan(args) -> dict:
    runtime = HeadlessRuntime()
    app = runtime.bootstrap(include_engine=False)
    project_folder = _abs_path(args.project)
    epub_path = _abs_path(args.epub)
    pm = _project_manager(project_folder)
    chapters = select_chapters(
        epub_path,
        pm,
        mode=args.chapters,
        patterns=args.chapter or [],
        offset=args.offset,
        limit=args.limit,
    )
    settings = build_session_settings(app.settings_manager, pm, chapters, args)
    plan = build_task_plan(epub_path, chapters, settings, pm)
    payload = {
        "ok": True,
        "epub": epub_path,
        "project": project_folder,
        "plan": plan.summary,
        "settings": _safe_settings_for_output(settings),
    }
    runtime.shutdown()
    return payload


def command_translate(args) -> dict:
    runtime = HeadlessRuntime()
    app = runtime.bootstrap(include_engine=True)
    project_folder = _abs_path(args.project)
    epub_path = _abs_path(args.epub)
    pm = _project_manager(project_folder)
    chapters = select_chapters(
        epub_path,
        pm,
        mode=args.chapters,
        patterns=args.chapter or [],
        offset=args.offset,
        limit=args.limit,
    )
    settings = build_session_settings(app.settings_manager, pm, chapters, args)
    plan = build_task_plan(epub_path, chapters, settings, pm)

    if not plan.payloads:
        runtime.shutdown()
        return {
            "ok": True,
            "status": "no_tasks",
            "epub": epub_path,
            "project": project_folder,
            "plan": plan.summary,
        }

    if plan.task_chains:
        app.task_manager.set_pending_task_chains(plan.task_chains)
    else:
        app.task_manager.set_pending_tasks(plan.payloads)

    observer = CliSessionObserver(
        app,
        verbose=bool(args.verbose),
        timeout_sec=args.timeout,
    )

    def start_session():
        app.event_bus.event_posted.emit({
            "event": "start_session_requested",
            "source": "cli",
            "data": {"settings": settings},
        })

    app.event_bus.set_data("cli_session_active", True)
    runtime.app_main.QtCore.QTimer.singleShot(0, start_session)
    app.exec()
    app.event_bus.pop_data("cli_session_active", None)

    result = observer.result_payload(app.task_manager)
    payload = {
        "ok": bool(result["finished"] and not result["timed_out"]),
        "status": "finished" if result["finished"] else "stopped",
        "epub": epub_path,
        "project": project_folder,
        "plan": plan.summary,
        "result": result,
    }
    runtime.shutdown()
    return payload


def _choose_translation_rel_path(versions: dict, suffix: str | None = None) -> str | None:
    if not isinstance(versions, dict) or not versions:
        return None
    if suffix is not None:
        return versions.get(suffix)
    for preferred_suffix in ("_validated.html", ""):
        if versions.get(preferred_suffix):
            return versions[preferred_suffix]
    for version_suffix, rel_path in versions.items():
        if version_suffix != "filtered" and rel_path:
            return rel_path
    return next(iter(versions.values()), None)


def _resolve_build_suffix(args) -> str | None:
    if getattr(args, "suffix", None):
        return args.suffix
    if getattr(args, "provider", None):
        api_config = _ensure_api_config_initialized()
        provider = api_config.api_providers().get(args.provider)
        if not provider:
            raise CliError(f"Unknown provider: {args.provider}")
        return provider.get("file_suffix")
    return None


def command_build_epub(args) -> dict:
    from .utils.epub_tools import EpubUpdater

    project_folder = _abs_path(args.project)
    epub_path = _abs_path(args.epub)
    output_path = _abs_path(args.output) if args.output else str(Path(epub_path).with_name(f"{Path(epub_path).stem}_translated.epub"))
    pm = _project_manager(project_folder)
    suffix = _resolve_build_suffix(args)
    chapters = select_chapters(
        epub_path,
        pm,
        mode="all",
        patterns=args.chapter or [],
        offset=args.offset,
        limit=args.limit,
    )

    updater = EpubUpdater(epub_path)
    selected = []
    missing = []
    for chapter in chapters:
        versions = pm.get_versions_for_original(chapter)
        rel_path = _choose_translation_rel_path(versions, suffix)
        if not rel_path:
            missing.append(chapter)
            continue
        full_path = os.path.join(project_folder, rel_path.replace("/", os.sep))
        if not os.path.exists(full_path):
            missing.append(chapter)
            continue
        updater.add_replacement(chapter, full_path)
        selected.append({"chapter": chapter, "file": full_path})

    if args.strict and missing:
        raise CliError(
            "Not all chapters have selected translated files.",
            payload={"missing": missing[:50], "missing_count": len(missing)},
        )
    if not selected:
        raise CliError("No translated chapters found for EPUB build.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    updater.update_and_save(output_path)
    return {
        "ok": True,
        "epub": epub_path,
        "project": project_folder,
        "output": output_path,
        "suffix": suffix,
        "replaced_count": len(selected),
        "missing_count": len(missing),
        "missing": missing[:50],
    }


def _chapter_order_key(rel_path: str) -> tuple:
    """Sort chapters the way the book reads: numerically when the file name
    carries a chapter number, alphabetically otherwise."""
    name = os.path.basename(str(rel_path))
    digits = re.findall(r"\d+", name)
    if digits:
        return (0, int(digits[-1]), name)
    return (1, 0, name)


def _chapter_number(rel_path: str) -> int | None:
    digits = re.findall(r"\d+", os.path.basename(str(rel_path)))
    return int(digits[-1]) if digits else None


def _count_terms_ahead(epub_path: str, terms: list[str], processed: list[str],
                       limit: int = 0) -> dict[str, dict]:
    """Count occurrences of each term in the chapters that come AFTER the
    processed ones.

    This is the mechanical answer to "will this term come back?" — the model
    only ever sees the current package, so recurrence cannot be judged from
    the prompt. Here the rest of the source book is actually scanned.
    """
    if not terms:
        return {}

    processed_set = {str(p) for p in processed}
    numbers = [n for n in (_chapter_number(p) for p in processed_set) if n is not None]
    last_processed = max(numbers) if numbers else None

    with zipfile.ZipFile(epub_path) as zf:
        candidates = [
            n for n in zf.namelist()
            if n.lower().endswith((".xhtml", ".html", ".htm"))
        ]
        candidates.sort(key=_chapter_order_key)

        ahead = []
        for name in candidates:
            if name in processed_set:
                continue
            num = _chapter_number(name)
            if last_processed is not None:
                if num is None or num <= last_processed:
                    continue
            ahead.append(name)
        if limit and limit > 0:
            ahead = ahead[:limit]

        stats = {t: {"count": 0, "chapters": []} for t in terms}
        for name in ahead:
            try:
                raw = zf.read(name).decode("utf-8", "replace")
            except Exception:
                continue
            for term in terms:
                hits = raw.count(term)
                if hits:
                    stats[term]["count"] += hits
                    stats[term]["chapters"].append(os.path.basename(name))

    for term in stats:
        stats[term]["scanned_chapters"] = len(ahead)
    return stats


def _glossary_entries_as_list(glossary: dict) -> list:
    """Convert the {original: {rus, note}} shape into the list shape the
    glossary engine expects for initial_glossary_list."""
    out = []
    for original, entry in (glossary or {}).items():
        if not original:
            continue
        out.append({
            "original": str(original),
            "rus": str((entry or {}).get("rus") or ""),
            "note": str((entry or {}).get("note") or ""),
        })
    return out


def _merge_into_authoritative(path: str, new_terms: list) -> dict:
    """Supplement-merge new terms into the authoritative glossary JSON.

    Existing entries always win: this file carries hand-made decisions and the
    generator must never silently overwrite them. Returns merge stats.
    """
    existing = _load_json_file(path, default=[])
    if not isinstance(existing, list):
        raise CliError(f"Authoritative glossary must be a JSON list: {path}")

    known = {
        str(e.get("original", "")).strip().lower()
        for e in existing if isinstance(e, dict) and e.get("original")
    }
    added = []
    for term in new_terms:
        if not isinstance(term, dict):
            continue
        original = str(term.get("original") or "").strip()
        key = original.lower()
        if not original or key in known:
            continue
        known.add(key)
        added.append({
            "original": original,
            "rus": str(term.get("rus") or ""),
            "note": str(term.get("note") or ""),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        })

    if added:
        backup = f"{path}.bak-glossary-{int(time.time())}"
        shutil.copy2(path, backup)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(existing + added, fh, ensure_ascii=False, indent=2)
    else:
        backup = None

    return {
        "existing": len(existing),
        "added": len(added),
        "total": len(existing) + len(added),
        "backup": backup,
        "added_terms": added,
    }


def command_glossary(args) -> dict:
    """Extract new glossary terms from chapters using the program's own
    glossary engine (glossary_batch_task), then optionally supplement the
    authoritative glossary file.

    This reuses the existing engine rather than reimplementing extraction:
    the same GlossaryBatchProcessor, prompt builder and SQLite store the GUI
    uses. The CLI only had no entry point for it.
    """
    runtime = HeadlessRuntime()
    app = runtime.bootstrap(include_engine=True)
    try:
        project_folder = _abs_path(args.project)
        epub_path = _abs_path(args.epub)
        pm = _project_manager(project_folder)
        chapters = select_chapters(
            epub_path,
            pm,
            mode=args.chapters,
            patterns=args.chapter or [],
            offset=args.offset,
            limit=args.limit,
        )
        if not chapters:
            runtime.shutdown()
            return {"ok": True, "status": "no_chapters", "epub": epub_path}

        settings = build_session_settings(app.settings_manager, pm, chapters, args)

        # Glossary generation prompt: the dedicated generator prompt, not the
        # translation prompt. --prompt-file would have been consumed as the
        # translation prompt by build_session_settings, so read it separately.
        gen_prompt = _load_text_file(getattr(args, "glossary_prompt_file", None))
        if gen_prompt is None:
            api_config = _ensure_api_config_initialized()
            gen_prompt = api_config.default_glossary_prompt()
        if not gen_prompt:
            raise CliError("No glossary generation prompt available.")

        # Conservative gate appended last so it overrides earlier encouragement
        # to produce entries.
        admission = _load_text_file(getattr(args, "admission_file", None))
        if admission:
            gen_prompt = f"{gen_prompt.rstrip()}\n\n{admission.strip()}\n"

        merge_mode = args.merge_mode or "supplement"
        settings.update({
            "glossary_generation_prompt": gen_prompt,
            "glossary_merge_mode": merge_mode,
            "initial_glossary_list": _glossary_entries_as_list(
                settings.get("full_glossary_data") or {}),
            "new_terms_limit": int(args.new_terms_limit or 0),
            "send_notes_in_sequence": True,
        })

        plan = build_task_plan(epub_path, chapters, settings, pm)
        if not plan.payloads:
            runtime.shutdown()
            return {"ok": True, "status": "no_tasks", "epub": epub_path,
                    "plan": plan.summary}

        # Rewrite translation payloads into glossary tasks, exactly as the GUI
        # does in ai_generation.py.
        glossary_tasks = []
        for payload in plan.payloads:
            task_type = payload[0]
            if task_type == "epub":
                glossary_tasks.append(
                    ("glossary_batch_task", payload[1], (payload[2],), {}))
            elif task_type == "epub_batch":
                glossary_tasks.append(
                    ("glossary_batch_task", payload[1], payload[2], {}))
        if not glossary_tasks:
            runtime.shutdown()
            return {"ok": False, "status": "unsupported_task_shape",
                    "detail": sorted({p[0] for p in plan.payloads})}

        app.task_manager.set_pending_tasks(glossary_tasks)

        observer = CliSessionObserver(
            app, verbose=bool(args.verbose), timeout_sec=args.timeout)

        def start_session():
            app.event_bus.event_posted.emit({
                "event": "start_session_requested",
                "source": "cli",
                "data": {"settings": settings},
            })

        app.event_bus.set_data("cli_session_active", True)
        runtime.app_main.QtCore.QTimer.singleShot(0, start_session)
        app.exec()
        app.event_bus.pop_data("cli_session_active", None)

        result = observer.result_payload(app.task_manager)

        # Terms landed in the project SQLite store; read them back.
        try:
            produced = app.task_manager.fetch_and_clean_glossary(mode=merge_mode)
        except Exception as exc:
            produced = []
            result["glossary_read_error"] = str(exc)

        known_before = {
            str(k).strip().lower()
            for k in (settings.get("full_glossary_data") or {})
        }
        fresh = [t for t in produced
                 if str(t.get("original", "")).strip().lower() not in known_before]

        # --- Recurrence gate, measured against the rest of the book ---------
        # The model only ever sees the current package, so it cannot know
        # whether a term comes back. Scan the following chapters and keep only
        # terms that actually reappear.
        min_ahead = int(getattr(args, "min_ahead", 0) or 0)
        ahead_stats = {}
        accepted, rejected = fresh, []
        if min_ahead > 0 and fresh:
            originals = [str(t.get("original") or "").strip()
                         for t in fresh if t.get("original")]
            ahead_stats = _count_terms_ahead(
                epub_path, originals, chapters,
                limit=int(getattr(args, "ahead_limit", 0) or 0),
            )
            accepted, rejected = [], []
            for term in fresh:
                original = str(term.get("original") or "").strip()
                stat = ahead_stats.get(original, {})
                hits = int(stat.get("count", 0))
                term["ahead_count"] = hits
                term["ahead_chapters"] = stat.get("chapters", [])[:8]
                (accepted if hits >= min_ahead else rejected).append(term)

        payload = {
            "ok": bool(result["finished"] and not result["timed_out"]),
            "status": "finished" if result["finished"] else "stopped",
            "epub": epub_path,
            "project": project_folder,
            "merge_mode": merge_mode,
            "chapters": len(chapters),
            "tasks": len(glossary_tasks),
            "terms_in_store": len(produced),
            "new_terms": len(fresh),
            "min_ahead": min_ahead,
            "scanned_ahead": (next(iter(ahead_stats.values()), {})
                              .get("scanned_chapters", 0) if ahead_stats else 0),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "terms": accepted,
            "rejected_terms": rejected,
            "plan": plan.summary,
            "result": result,
        }

        export_to = getattr(args, "export_to", None)
        if export_to and accepted:
            payload["export"] = _merge_into_authoritative(_abs_path(export_to), accepted)
        elif export_to:
            payload["export"] = {"added": 0, "note": "no terms passed the recurrence gate"}

        runtime.shutdown()
        return payload
    except BaseException:
        runtime.shutdown()
        raise


def command_review(args) -> dict:
    """Verify finished translations of a package against the CN source.

    Reads the translated HTML that `translate` wrote into the project folder,
    checks each chapter against the original with a dedicated review prompt, and
    writes a corrected copy next to it. Nothing is inserted into any EPUB here --
    that stays a separate, explicit step.
    """
    from . import review as review_mod

    runtime = HeadlessRuntime()
    app = runtime.bootstrap(include_engine=False)
    try:
        project_folder = _abs_path(args.project)
        epub_path = _abs_path(args.epub)
        pm = _project_manager(project_folder)

        chapters = select_chapters(
            epub_path,
            pm,
            mode=args.chapters,
            patterns=args.chapter or [],
            offset=args.offset,
            limit=args.limit,
        )
        if not chapters:
            raise CliError("No chapters selected for review.")

        prompt_template = _load_text_file(args.prompt_file)
        if not prompt_template:
            raise CliError("--prompt-file is required for review.")
        for token in ("{source}", "{text}"):
            if token not in prompt_template:
                raise CliError(f"Review prompt is missing the {token} placeholder.")

        # Optional per-novel rules injected into the universal prompt's {rules}
        # block. Keeps the prompt itself language- and book-agnostic.
        project_rules = _load_text_file(args.rules_file) or ""

        glossary = load_project_glossary(project_folder, args.glossary)

        api_config = _ensure_api_config_initialized()
        saved_settings = app.settings_manager.load_settings() or {}
        provider_id = _resolve_provider(api_config, saved_settings, args.provider)
        model_name, model_config = _resolve_model(
            api_config, provider_id, saved_settings, args.model
        )
        provider_conf = api_config.api_providers().get(provider_id) or {}
        endpoint = review_mod.resolve_endpoint(provider_id, provider_conf)
        keys = _resolve_api_keys(
            api_config, app.settings_manager, provider_id, saved_settings, args
        )
        headers = review_mod.build_headers(provider_id, keys[0] if keys else None)
        model_id = model_config.get("id") or model_name

        suffix = _resolve_build_suffix(args) or provider_conf.get("file_suffix")

        results = []
        for chapter_rel in chapters:
            # Same lookup build_epub uses, so review reads exactly the file that
            # assembly would pick up.
            versions = pm.get_versions_for_original(chapter_rel)
            rel_path = _choose_translation_rel_path(versions, suffix)
            if not rel_path:
                rel_path = _choose_translation_rel_path(versions)
            if not rel_path:
                results.append({"chapter": chapter_rel, "ok": False,
                                "status": "no_translation"})
                continue

            src_file = os.path.join(project_folder, rel_path)
            if not os.path.exists(src_file):
                results.append({"chapter": chapter_rel, "ok": False,
                                "status": "missing_file", "path": src_file})
                continue

            translated_html = Path(src_file).read_text(encoding="utf-8")
            res = review_mod.review_chapter(
                cn_epub=epub_path,
                chapter_rel=chapter_rel,
                translated_html=translated_html,
                glossary=glossary,
                prompt_template=prompt_template,
                endpoint=endpoint,
                headers=headers,
                model_id=model_id,
                project_rules=project_rules,
                timeout=args.timeout or review_mod.DEFAULT_TIMEOUT,
            )

            if res.get("ok") and res.get("html"):
                if args.apply:
                    # Overwrite the translation in place. This is what makes the
                    # reviewed text reach the EPUB: add_new_chapters.py discovers
                    # files by chap_<n>_translated_<suffix>.html, and \w+ also
                    # matches "gl_reviewed", so leaving a second candidate in the
                    # insertion folder would let os.listdir() order decide which
                    # version lands in the book.
                    #
                    # The backup must live OUTSIDE that folder for the same
                    # reason: that script uses re.match, which is not anchored at
                    # the end, so "chap_924_translated_gl.html.prereview" still
                    # matches the pattern and would itself become a rival
                    # candidate. A sibling directory keeps discovery unambiguous.
                    backup_dir = os.path.join(project_folder, "prereview_backups")
                    os.makedirs(backup_dir, exist_ok=True)
                    backup = os.path.join(
                        backup_dir, os.path.basename(src_file) + ".orig")
                    if not os.path.exists(backup):
                        Path(backup).write_text(translated_html, encoding="utf-8")
                    Path(src_file).write_text(res["html"], encoding="utf-8")
                    res["applied"] = True
                    res["backup"] = backup
                    res["output"] = src_file
                else:
                    stem, ext = os.path.splitext(src_file)
                    out_file = f"{stem}_reviewed{ext or '.html'}"
                    Path(out_file).write_text(res["html"], encoding="utf-8")
                    res["applied"] = False
                    res["output"] = out_file
                res["source_file"] = src_file
            res.pop("html", None)
            results.append(res)

        ok_count = sum(1 for r in results if r.get("ok"))
        changed = sum(1 for r in results if r.get("status") == "changed")
        return {
            "ok": ok_count == len(results),
            "epub": epub_path,
            "project": project_folder,
            "provider": provider_id,
            "model": model_name,
            "applied": bool(args.apply),
            "reviewed": ok_count,
            "changed": changed,
            "total_edits": sum(r.get("edit_count", 0) for r in results),
            "total": len(results),
            "results": results,
        }
    finally:
        runtime.shutdown()


def _add_common_project_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epub", required=True, help="Source EPUB path.")
    parser.add_argument("--project", required=True, help="Translator project/output folder.")
    parser.add_argument("--chapters", choices=["pending", "all", "translated"], default="pending")
    parser.add_argument("--chapter", action="append", help="Chapter glob/substr filter. Can be repeated.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", help="Provider id, e.g. gemini, deepseek, openrouter, local.")
    parser.add_argument("--model", help="Model display name from config/api_providers.json.")
    parser.add_argument("--api-key", action="append", help="API key/session token. Can be repeated.")
    parser.add_argument("--api-key-file", help="Text file with one API key per line.")
    parser.add_argument("--all-keys", action="store_true", help="Use all configured keys for the provider.")
    parser.add_argument("--workers", type=int, help="Number of parallel workers.")
    parser.add_argument("--rpm", type=int, help="Per-worker/request pool RPM limit override.")
    parser.add_argument("--temperature", type=float, help="Temperature override.")
    parser.add_argument("--mode", choices=["saved", "single", "batch", "chunk", "sequential"], default="saved")
    parser.add_argument("--task-size", type=int, help="Input char limit for batch/chunk task building.")
    parser.add_argument("--splits", type=int, default=1, help="Sequential mode chain count.")
    parser.add_argument("--force-accept", action="store_true", help="Skip HTML validation rejection.")
    parser.add_argument("--json-epub", action="store_true", help="Use JSON EPUB transport pipeline.")
    parser.add_argument("--prompt-file", help="Custom translation prompt file.")
    parser.add_argument("--glossary", help="Glossary JSON path. Defaults to <project>/project_glossary.json.")
    parser.add_argument("--settings-json", help="JSON file with final settings overrides.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translator-cli",
        description="Headless CLI tools for the EPUB translator.",
    )
    parser.add_argument("--compact", action="store_true", help="Print compact JSON.")
    parser.add_argument("--debug", action="store_true", help="Include traceback in JSON errors.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Show providers, keys, saved settings, and optional project status.")
    status.add_argument("--project")
    status.add_argument("--epub")
    status.set_defaults(func=command_status)

    plan = subparsers.add_parser("plan", help="Build a translation task plan without calling APIs.")
    _add_common_project_args(plan)
    _add_common_run_args(plan)
    plan.set_defaults(func=command_plan)

    translate = subparsers.add_parser("translate", help="Run a full headless translation session.")
    _add_common_project_args(translate)
    _add_common_run_args(translate)
    translate.add_argument("--timeout", type=int, help="Stop the session after N seconds.")
    translate.add_argument("--verbose", action="store_true", help="Mirror app log messages to stderr.")
    translate.set_defaults(func=command_translate)

    review = subparsers.add_parser("review", help="Verify finished translations against the CN source and write corrected copies.")
    _add_common_project_args(review)
    _add_common_run_args(review)
    review.add_argument("--suffix", help="Explicit translation_map suffix to review.")
    review.add_argument("--rules-file", help="Optional per-novel rules file injected into the prompt's {rules} block. Keeps the review prompt itself book-agnostic.")
    review.add_argument("--timeout", type=int, help="Per-chapter API timeout in seconds.")
    review.add_argument("--apply", action="store_true", help="Overwrite the translated file in place (keeps a .prereview backup) so the reviewed text is what gets inserted. Without this, writes a separate _reviewed.html for inspection.")
    review.set_defaults(func=command_review)

    glossary = subparsers.add_parser("glossary", help="Extract new glossary terms from chapters using the program's glossary engine, and optionally supplement the authoritative glossary file.")
    _add_common_project_args(glossary)
    _add_common_run_args(glossary)
    glossary.add_argument("--glossary-prompt-file", help="Glossary generation prompt (defaults to config/default_glossary_prompt.txt). This is NOT the translation prompt.")
    glossary.add_argument("--admission-file", help="Extra gatekeeping block appended to the generation prompt so terms are added only as a last resort (e.g. config/glossary_admission_threshold.txt).")
    glossary.add_argument("--merge-mode", choices=["supplement", "update", "accumulate"], default="supplement", help="supplement (default): existing entries always win, only genuinely new terms are kept.")
    glossary.add_argument("--new-terms-limit", type=int, default=0, help="Hard cap on new terms per task; 0 disables the cap.")
    glossary.add_argument("--export-to", help="Authoritative glossary JSON to supplement with the new terms (a timestamped .bak is written first). Without this flag nothing outside the project folder is touched.")
    glossary.add_argument("--min-ahead", type=int, default=0, help="Recurrence gate: keep a new term only if it occurs at least N times in the chapters AFTER the processed ones. 0 disables the gate. The model only sees the current package, so recurrence is measured here, not guessed by the model.")
    glossary.add_argument("--ahead-limit", type=int, default=0, help="Scan at most N following chapters for the recurrence gate; 0 scans to the end of the book.")
    glossary.add_argument("--timeout", type=int, help="Session timeout in seconds.")
    glossary.add_argument("--verbose", action="store_true", help="Mirror app log messages to stderr.")
    glossary.set_defaults(func=command_glossary)

    def command_harvest_terms(args) -> dict:
        """Collect <!--NEW_TERMS [...]--> tails the translator appended after
        </html> during a translate run, strip them from the translated HTML
        files, and optionally supplement the authoritative glossary.

        The tail format is defined by the translation prompt (see the
        GLOSSARY FEEDBACK section), so this command needs no API access: it
        only reads what the model already wrote. Recurrence gating reuses the
        same --min-ahead mechanics as the glossary subcommand: a term that
        never reappears in the following chapters is rejected, because the
        model cannot know that from its fragment alone.
        """
        project_folder = _abs_path(args.project)
        epub_path = _abs_path(args.epub)
        if not os.path.isdir(project_folder):
            raise CliError(f"Project folder not found: {project_folder}")

        tail_re = re.compile(
            r"<!--NEW_TERMS\s*(\[.*?\])\s*-->\s*$", re.DOTALL
        )
        scan_root = project_folder
        harvested: dict[str, list] = {}
        cleaned_files: list[str] = []
        parse_errors: list[dict] = []

        for dirpath, _dirnames, filenames in os.walk(scan_root):
            for fname in filenames:
                if "_translated_" not in fname or not fname.endswith((".html", ".xhtml")):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        content = fh.read()
                except OSError as exc:
                    parse_errors.append({"file": fname, "error": str(exc)})
                    continue
                m = tail_re.search(content.rstrip() + "\n")
                if not m:
                    continue
                rel = os.path.relpath(fpath, project_folder)
                try:
                    terms = json.loads(m.group(1))
                    if not isinstance(terms, list):
                        raise ValueError("not a JSON array")
                except (ValueError, json.JSONDecodeError) as exc:
                    parse_errors.append({"file": rel, "error": f"bad NEW_TERMS JSON: {exc}"})
                    # Strip the malformed tail anyway so it cannot leak into the EPUB.
                    cleaned = content[: m.start()].rstrip() + "\n"
                    with open(fpath, "w", encoding="utf-8") as fh:
                        fh.write(cleaned)
                    cleaned_files.append(rel)
                    continue
                chapter_terms = []
                for t in terms:
                    if isinstance(t, dict) and str(t.get("original") or "").strip():
                        chapter_terms.append({
                            "original": str(t["original"]).strip(),
                            "rus": str(t.get("rus") or "").strip(),
                            "note": str(t.get("note") or "").strip(),
                        })
                if chapter_terms:
                    harvested[rel] = chapter_terms
                cleaned = content[: m.start()].rstrip() + "\n"
                with open(fpath, "w", encoding="utf-8") as fh:
                    fh.write(cleaned)
                cleaned_files.append(rel)

        # Flatten in stable file order, dropping duplicates by original key.
        flat: list[dict] = []
        seen: set[str] = set()
        for rel in sorted(harvested):
            for t in harvested[rel]:
                key = t["original"].lower()
                if key in seen:
                    continue
                seen.add(key)
                t = dict(t)
                t["source_file"] = rel
                flat.append(t)

        # Recurrence gate against the rest of the book (same logic as glossary).
        min_ahead = int(getattr(args, "min_ahead", 0) or 0)
        ahead_stats: dict = {}
        accepted, rejected = flat, []
        if min_ahead > 0 and flat:
            processed = list(harvested.keys())
            originals = [t["original"] for t in flat]
            ahead_stats = _count_terms_ahead(
                epub_path, originals, processed,
                limit=int(getattr(args, "ahead_limit", 0) or 0),
            )
            accepted, rejected = [], []
            for term in flat:
                stat = ahead_stats.get(term["original"], {})
                hits = int(stat.get("count", 0))
                term["ahead_count"] = hits
                term["ahead_chapters"] = stat.get("chapters", [])[:8]
                (accepted if hits >= min_ahead else rejected).append(term)

        export_to = getattr(args, "export_to", None)
        export_result = None
        if export_to and accepted:
            export_result = _merge_into_authoritative(_abs_path(export_to), accepted)
        elif export_to:
            export_result = {"added": 0, "note": "no terms passed the recurrence gate"}

        return {
            "ok": not parse_errors,
            "epub": epub_path,
            "project": project_folder,
            "scanned_translated_files": len(cleaned_files),
            "files_with_terms": len(harvested),
            "cleaned_files": cleaned_files,
            "new_terms": len(flat),
            "terms": flat,
            "min_ahead": min_ahead,
            "scanned_ahead": (next(iter(ahead_stats.values()), {})
                              .get("scanned_chapters", 0) if ahead_stats else 0),
            "accepted": len(accepted),
            "rejected_terms": rejected,
            "export": export_result,
            "parse_errors": parse_errors,
        }

    harvest = subparsers.add_parser("harvest-terms", help="Collect <!--NEW_TERMS [...]--> tails appended by the translator after </html>, strip them from translated HTML files, and optionally supplement the authoritative glossary. No API calls.")
    harvest.add_argument("--epub", required=True, help="Source EPUB, used for the recurrence gate (chapters after the harvested ones).")
    harvest.add_argument("--project", required=True)
    harvest.add_argument("--export-to", help="Authoritative glossary JSON to supplement with accepted terms (timestamped .bak first). Without this flag the harvest is report-only.")
    harvest.add_argument("--min-ahead", type=int, default=0, help="Recurrence gate: keep a new term only if it occurs at least N times in the chapters AFTER the harvested ones. 0 disables the gate.")
    harvest.add_argument("--ahead-limit", type=int, default=0, help="Scan at most N following chapters for the recurrence gate; 0 scans to the end of the book.")
    harvest.set_defaults(func=command_harvest_terms)

    build_epub = subparsers.add_parser("build-epub", help="Build an EPUB by replacing source chapters with translated files.")
    build_epub.add_argument("--epub", required=True)
    build_epub.add_argument("--project", required=True)
    build_epub.add_argument("--output")
    build_epub.add_argument("--provider", help="Use this provider's file suffix.")
    build_epub.add_argument("--suffix", help="Explicit translation_map suffix to select.")
    build_epub.add_argument("--chapter", action="append")
    build_epub.add_argument("--offset", type=int, default=0)
    build_epub.add_argument("--limit", type=int)
    build_epub.add_argument("--strict", action="store_true", help="Fail if any selected source chapter has no translation.")
    build_epub.set_defaults(func=command_build_epub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        with _redirect_internal_stdout_to_stderr():
            payload = args.func(args)
        _write_json(payload, pretty=not args.compact)
        return 0 if payload.get("ok", True) else 1
    except CliError as exc:
        payload = {"ok": False, "error": str(exc)}
        payload.update(exc.payload)
        if getattr(args, "debug", False):
            payload["traceback"] = traceback.format_exc()
        _write_json(payload, pretty=not args.compact)
        return exc.exit_code
    except Exception as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if getattr(args, "debug", False):
            payload["traceback"] = traceback.format_exc()
        _write_json(payload, pretty=not args.compact)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
