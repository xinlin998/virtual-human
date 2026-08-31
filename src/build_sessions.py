from typing import Any


def build_sessions(
        turns: list[dict[str,Any]],
        session_gap_seconds: int = 1800,
        max_session_duration_seconds: int = 7200,
        max_session_turns: int = 80
) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    if not turns:
        return [],[]

    if session_gap_seconds <= 0:
        raise ValueError("session_gap_seconds必须大于0")

    if max_session_duration_seconds <= 0:
        raise ValueError("max_session_duration_seconds必须大于0")

    if max_session_turns <= 0:
        raise ValueError("max_session_turns必须大于0")

    sorted_turns = sorted(
        turns,
        key=lambda turn: (
            int(turn["start_timestamp"]),
            str(turn["turn_id"])
        )
    )

    sessions = []
    current_turns = []

    for turn in sorted_turns:
        turn["start_timestamp"] = int(turn["start_timestamp"])
        turn["end_timestamp"] = int(turn["end_timestamp"])

        if not current_turns:
            current_turns.append(turn)
            continue

        previous_turn = current_turns[-1]

        gap = (
            turn["start_timestamp"]
            - previous_turn["end_timestamp"]
        )

        session_duration = (
            turn["end_timestamp"]
            - current_turns[0]["start_timestamp"]
        )

        should_split = (
            gap > session_gap_seconds
            or session_duration > max_session_duration_seconds
            or len(current_turns) >= max_session_turns
        )

        if should_split:
            session = _build_session(
                session_index=len(sessions),
                turns=current_turns
            )

            sessions.append(session)
            current_turns = [turn]
        else:
            current_turns.append(turn)

    if current_turns:
        session = _build_session(
            session_index=len(sessions),
            turns=current_turns
        )

        sessions.append(session)

    print(f"Turn总数：{len(sorted_turns)}")
    print(f"Session数量：{len(sessions)}")

    if sessions:
        turn_counts = [
            session["turn_count"]
            for session in sessions
        ]

        durations = [
            session["duration_seconds"]
            for session in sessions
        ]

        print(f"平均每个Session的Turn数：{sum(turn_counts) / len(turn_counts):.2f}")
        print(f"最大Session Turn数：{max(turn_counts)}")
        print(f"平均Session时长：{sum(durations) / len(durations):.2f}秒")
        print(f"最长Session时长：{max(durations)}秒")

    return sessions,sorted_turns


def _build_session(
        session_index: int,
        turns: list[dict[str,Any]]
) -> dict[str,Any]:
    session_id = f"session_{session_index:07d}"

    for turn in turns:
        turn["session_id"] = session_id

    start_timestamp = turns[0]["start_timestamp"]
    end_timestamp = turns[-1]["end_timestamp"]

    return {
        "session_id":session_id,
        "start_timestamp":start_timestamp,
        "end_timestamp":end_timestamp,
        "duration_seconds":end_timestamp - start_timestamp,
        "turn_count":len(turns),
        "turn_ids":[turn["turn_id"] for turn in turns],
        "speaker_sequence":[turn["speaker"] for turn in turns],
        "message_count":sum(int(turn["message_count"]) for turn in turns),
        "turns":turns
    }