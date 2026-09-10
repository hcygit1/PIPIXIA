"""File-system lifecycle for offline Skill candidates and active releases."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class SkillVersion:
    skill_id: str
    version: int
    status: str
    parent_version: int | None = None
    source: str = ""
    reason: str = ""


class SkillVersionStore:
    """Keeps candidate artifacts out of the directory scanned by live agents."""

    def __init__(self, *, evolution_root: Path, active_root: Path):
        self.evolution_root = evolution_root
        self.active_root = active_root

    def candidate_dir(self, skill_id: str, version: int) -> Path:
        return self.evolution_root / skill_id / "candidates" / f"v{version}"

    def active_version(self, skill_id: str) -> int | None:
        manifest = self._read_json(self.active_root / skill_id / "manifest.json")
        value = manifest.get("active_version")
        return int(value) if isinstance(value, int) or (isinstance(value, str) and value.isdigit()) else None

    def next_candidate_version(self, skill_id: str) -> int:
        versions = [self.active_version(skill_id) or 0]
        candidates = self.evolution_root / skill_id / "candidates"
        if candidates.is_dir():
            for path in candidates.iterdir():
                if path.is_dir() and path.name.startswith("v") and path.name[1:].isdigit():
                    versions.append(int(path.name[1:]))
        return max(versions) + 1

    def create_candidate(
        self,
        *,
        skill_id: str,
        content: str,
        source: str,
        reason: str,
        parent_version: int | None = None,
    ) -> SkillVersion:
        version = self.next_candidate_version(skill_id)
        candidate = SkillVersion(
            skill_id=skill_id,
            version=version,
            status="candidate",
            parent_version=parent_version if parent_version is not None else self.active_version(skill_id),
            source=source,
            reason=reason,
        )
        destination = self.candidate_dir(skill_id, version)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "SKILL.md").write_text(content, encoding="utf-8")
        self._write_metadata(destination / "candidate.json", candidate)
        return candidate

    def set_candidate_status(self, candidate: SkillVersion, status: str) -> SkillVersion:
        if status not in {"accepted", "rejected", "needs_eval_fix", "external_failure", "pending_regression", "abandoned"}:
            raise ValueError(f"unsupported candidate status: {status}")
        updated = SkillVersion(**{**asdict(candidate), "status": status})
        self._write_metadata(
            self.candidate_dir(updated.skill_id, updated.version) / "candidate.json", updated,
        )
        return updated

    def publish(self, candidate: SkillVersion) -> None:
        if candidate.status != "accepted":
            raise ValueError("only accepted candidates can be published")
        source = self.candidate_dir(candidate.skill_id, candidate.version) / "SKILL.md"
        if not source.is_file():
            raise FileNotFoundError(source)
        skill_root = self.active_root / candidate.skill_id
        manifest = self._read_json(skill_root / "manifest.json")
        version_dir = skill_root / "versions" / f"v{candidate.version}"
        version_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source, version_dir / "SKILL.md")
        skill_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, skill_root / "SKILL.md")
        history = list(manifest.get("history", [])) if isinstance(manifest.get("history", []), list) else []
        previous = manifest.get("active_version")
        if previous is not None and int(previous) != candidate.version:
            history.append(int(previous))
        self._write_json(skill_root / "manifest.json", {
            "skill_id": candidate.skill_id,
            "active_version": candidate.version,
            "status": "active",
            "history": history,
        })

    def rollback(self, skill_id: str, version: int | None = None) -> int:
        """Switch the live pointer to an already published version."""
        skill_root = self.active_root / skill_id
        manifest = self._read_json(skill_root / "manifest.json")
        current = int(manifest.get("active_version", 0) or 0)
        history = list(manifest.get("history", [])) if isinstance(manifest.get("history", []), list) else []
        target = version if version is not None else (int(history[-1]) if history else None)
        if target is None or target == current:
            raise ValueError("no previous active version available")
        source = skill_root / "versions" / f"v{target}" / "SKILL.md"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, skill_root / "SKILL.md")
        remaining = [item for item in history if int(item) != target]
        remaining.append(current)
        self._write_json(skill_root / "manifest.json", {
            "skill_id": skill_id, "active_version": target,
            "status": "active", "history": remaining,
        })
        return target

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _write_metadata(self, path: Path, candidate: SkillVersion) -> None:
        data = asdict(candidate)
        data["created_at"] = datetime.now(timezone.utc).isoformat()
        self._write_json(path, data)
