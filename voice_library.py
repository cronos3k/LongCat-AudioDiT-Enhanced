"""
Voice Library — persistent store of (name, audio_path, transcription) profiles.

Files live in ./voices/ (project-local, never Windows user dirs).
Index is ./voices/library.json.

Usage:
    lib = VoiceLibrary()
    lib.add("Alice", "/path/to/alice.wav", "Hello, my name is Alice.")
    voice = lib.get("Alice")
    names = lib.names()
    lib.remove("Alice")
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

VOICES_DIR   = Path(__file__).parent / "voices"
LIBRARY_FILE = VOICES_DIR / "library.json"


class VoiceLibrary:
    def __init__(self):
        VOICES_DIR.mkdir(exist_ok=True)
        self._data: dict = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if LIBRARY_FILE.exists():
            try:
                return json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"voices": {}}

    def _save(self):
        LIBRARY_FILE.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _reload(self):
        """Re-read from disk (picks up changes from other processes)."""
        self._data = self._load()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(self, name: str, audio_src: str, transcription: str) -> dict:
        """
        Add or overwrite a voice profile.

        The audio file is COPIED into ./voices/ so the library is self-contained.
        Returns the stored entry dict.
        """
        name = name.strip()
        if not name:
            raise ValueError("Voice name must not be empty.")

        # Copy audio into voices dir
        src = Path(audio_src)
        # Use a safe filename derived from the name + timestamp
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()
        dest = VOICES_DIR / f"{safe}_{int(time.time())}{src.suffix or '.wav'}"
        shutil.copy2(str(src), str(dest))

        entry = {
            "name":          name,
            "audio_path":    str(dest),
            "transcription": transcription.strip(),
            "added":         datetime.now().isoformat(timespec="seconds"),
        }
        self._reload()
        self._data["voices"][name] = entry
        self._save()
        return entry

    def get(self, name: str) -> Optional[dict]:
        """Return entry dict or None."""
        self._reload()
        return self._data["voices"].get(name)

    def names(self) -> list[str]:
        """Sorted list of voice names."""
        self._reload()
        return sorted(self._data["voices"].keys())

    def remove(self, name: str) -> bool:
        """Delete a voice profile (and its audio file). Returns True if it existed."""
        self._reload()
        entry = self._data["voices"].pop(name, None)
        if entry is None:
            return False
        audio = Path(entry.get("audio_path", ""))
        if audio.exists() and audio.parent == VOICES_DIR:
            audio.unlink(missing_ok=True)
        self._save()
        return True

    def all_entries(self) -> list[dict]:
        """All entries as a list, sorted by name."""
        self._reload()
        return sorted(self._data["voices"].values(), key=lambda e: e["name"].lower())

    def summary_text(self) -> str:
        entries = self.all_entries()
        if not entries:
            return "Voice library is empty. Save a voice to get started."
        lines = [f"{len(entries)} saved voice(s):"]
        for e in entries:
            lines.append(f"  • {e['name']}   —   \"{e['transcription'][:60]}{'…' if len(e['transcription'])>60 else ''}\"")
        return "\n".join(lines)


# Module-level singleton
_library: Optional[VoiceLibrary] = None

def get_library() -> VoiceLibrary:
    global _library
    if _library is None:
        _library = VoiceLibrary()
    return _library
