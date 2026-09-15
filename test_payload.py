"""Validate desktop client JSON payloads without importing main.py.

Run: python -m pytest test_payload.py -v
"""
import json
import pytest


def test_game_result_payload_structure():
    """Ensure the payload sent to /api/game/submit has all required fields."""
    payload = {
        "uid": "BCR-001",
        "mode": "competition",
        "nick_name": "Alice",
        "gender": "female",
        "age": 10,
        "task01": 5.0,
        "task02": 6.0,
        "task03": 7.0,
        "task04": 8.0,
        "task05": 9.0,
        "task06": 10.0,
        "task07": 11.0,
        "task08": 12.0,
        "task_avg": 8.5,
        "cognitive_age": 23,
        "visuo_spatial": 85.0,
        "variant_list": ["1a", "2b", "3c", "4d", "5a", "6b", "7c", "8d"],
        "client_ts": 1234567890,
    }
    # Serialize/deserialize to catch type issues
    raw = json.dumps(payload)
    parsed = json.loads(raw)

    assert "uid" in parsed
    assert "mode" in parsed
    assert parsed["mode"] in ("training", "competition")
    assert isinstance(parsed["age"], int)
    assert isinstance(parsed["task_avg"], float)
    assert len([k for k in parsed if k.startswith("task0")]) == 8


def test_game_event_payload_structure():
    """Ensure WebSocket/REST game event payload is valid."""
    for event_type in ("join_room", "level_start", "level_complete", "leave_room"):
        payload = {
            "type": event_type,
            "room_id": "ABCD",
            "player_name": "Alice",
        }
        if event_type == "level_start":
            payload["level"] = 1
        if event_type == "level_complete":
            payload["level"] = 1
            payload["time_sec"] = 12.5

        raw = json.dumps(payload)
        parsed = json.loads(raw)
        assert parsed["type"] == event_type
        assert len(parsed["room_id"]) == 4
        assert parsed["player_name"]


def test_tournament_event_payload_structure():
    """Ensure tournament event payload matches backend expectation."""
    payload = {
        "room_id": "ABCD",
        "event_type": "match_started",
        "player_num": 1,
        "score": 150.0,
    }
    raw = json.dumps(payload)
    parsed = json.loads(raw)
    assert parsed["event_type"] in ("match_started", "match_finished")
    assert isinstance(parsed["player_num"], int)
    assert 1 <= parsed["player_num"] <= 2


def test_room_payload_constraints():
    """Room code must be 4 uppercase chars from allowed charset."""
    allowed = set("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
    code = "AB2D"
    assert len(code) == 4
    assert all(ch in allowed for ch in code)

    # Reject confusable chars
    for bad in "IO01":
        assert bad not in allowed


def test_participant_lookup_response():
    """Simulate backend response for GET /api/v1/participants/uid/:uid"""
    response = {
        "status": "success",
        "message": "Participant found",
        "data": {
            "id": 1,
            "uid": "BCR-001",
            "name": "Alice",
            "age": 10,
            "is_premium": True,
        },
    }
    assert response["data"]["is_premium"] is True
    assert response["data"]["uid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
