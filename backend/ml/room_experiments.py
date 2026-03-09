from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ml.policy_config import RoomDiagnosticProfile, load_room_diagnostic_profiles


@dataclass(frozen=True)
class RoomReplayCandidate:
    profile_name: str
    room: str
    replay_mode: str
    description: str
    typed_policy: dict[str, Any]
    fast_replay_eligible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "room": self.room,
            "replay_mode": self.replay_mode,
            "description": self.description,
            "typed_policy": dict(self.typed_policy),
            "fast_replay_eligible": bool(self.fast_replay_eligible),
        }


def build_room_replay_candidates(
    profile_names: Iterable[str] | None = None,
) -> list[RoomReplayCandidate]:
    profiles = load_room_diagnostic_profiles()
    selected_names = [
        str(name).strip().lower()
        for name in (profile_names or profiles.keys())
        if str(name).strip()
    ]
    candidates: list[RoomReplayCandidate] = []
    for name in selected_names:
        profile = profiles.get(name)
        if profile is None:
            continue
        candidates.append(
            RoomReplayCandidate(
                profile_name=profile.name,
                room=profile.room,
                replay_mode=profile.replay_mode,
                description=profile.description,
                typed_policy=dict(profile.typed_policy),
                fast_replay_eligible=True,
            )
        )
    return candidates


def build_room_replay_report(
    profile_names: Iterable[str] | None = None,
    *,
    fast_replay_only: bool = False,
) -> dict[str, Any]:
    candidates = build_room_replay_candidates(profile_names)
    if fast_replay_only:
        candidates = [candidate for candidate in candidates if candidate.fast_replay_eligible]
    return {
        "status": "ok",
        "candidate_count": len(candidates),
        "fast_replay_only": bool(fast_replay_only),
        "profiles": [candidate.to_dict() for candidate in candidates],
    }
