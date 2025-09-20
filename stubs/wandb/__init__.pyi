from __future__ import annotations

from typing import Any

class Artifact:
    def __init__(
        self,
        name: str,
        type: str,
        description: str | None = ...,
        metadata: dict[str, object] | None = ...,
    ) -> None: ...
    def add_file(
        self,
        local_path: str,
        name: str | None = ...,
        root: str | None = ...,
        relpath: str | None = ...,
    ) -> None: ...
    def download(self) -> str: ...

class Run:
    def log_artifact(self, artifact: Artifact) -> None: ...
    def use_artifact(self, artifact_name: str) -> Artifact: ...
