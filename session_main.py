import json
from pathlib import Path
from typing import Any

from src.build_sessions import build_sessions
from src.config import load_config
from src.utils import ensure_parent_dir,save_jsonl,load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(relative_path: str) -> Path:
    return (PROJECT_ROOT / relative_path).resolve()



def main() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/pipeline.yaml"
    )

    data_config = config["data"]
    session_config = config["session"]

    turns_jsonl = _project_path(
        data_config["turns_jsonl"]
    )

    sessions_jsonl = _project_path(
        data_config["sessions_jsonl"]
    )

    turns_with_sessions_jsonl = _project_path(
        data_config["turns_with_sessions_jsonl"]
    )

    print(f"读取Turn文件：{turns_jsonl}")

    turns = load_jsonl(turns_jsonl)

    sessions,turns_with_sessions = build_sessions(
    turns=turns,
    session_gap_seconds=session_config["hard_gap_seconds"],
    max_session_duration_seconds=session_config["max_session_duration_seconds"],
    max_session_turns=session_config["max_session_turns"]
)

    ensure_parent_dir(sessions_jsonl)
    save_jsonl(
        sessions,
        sessions_jsonl
    )

    ensure_parent_dir(turns_with_sessions_jsonl)
    save_jsonl(
        turns_with_sessions,
        turns_with_sessions_jsonl
    )

    print("\nSession切分完成")
    print(f"Session文件：{sessions_jsonl}")
    print(f"回填session_id后的Turn文件：{turns_with_sessions_jsonl}")


if __name__ == "__main__":
    main()