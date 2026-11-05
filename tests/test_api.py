from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def new_session() -> str:
    r = client.post("/sessions")
    assert r.status_code == 200
    assert r.json()["model_status"] == "toy_untrained"
    return r.json()["session_id"]


def test_api_session_isolation_and_seen_item_filter():
    session_a, session_b = new_session(), new_session()
    baseline_b = client.get(f"/sessions/{session_b}/recommendations?variant=base").json()
    assert baseline_b["history_length"] == 0

    r = client.post(
        f"/sessions/{session_a}/ratings",
        json={"item_id": "demo-astro", "rating": 5},
    )
    assert r.status_code == 200
    assert r.json()["history_length"] == 1

    for variant in ("base", "vector", "matrix"):
        a = client.get(
            f"/sessions/{session_a}/recommendations", params={"variant": variant}
        ).json()
        assert a["history_length"] == 1
        assert "demo-astro" not in {x["item_id"] for x in a["recommendations"]}
        assert a["model_status"] == "toy_untrained"
    unchanged = client.get(f"/sessions/{session_b}/recommendations?variant=base").json()
    assert unchanged == baseline_b


def test_api_validation_replay_and_reset():
    s = new_session()
    assert client.post(
        f"/sessions/{s}/ratings",
        json={"item_id": "demo-astro", "rating": 6},
    ).status_code == 422
    assert client.post(
        f"/sessions/{s}/ratings",
        json={"item_id": "not-an-item", "rating": 5},
    ).status_code == 404

    payload = {"events": [
        {"item_id": "demo-astro", "rating": 5},
        {"item_id": "demo-rally", "rating": 2},
    ]}
    assert client.post(f"/sessions/{s}/replay", json=payload).json()["history_length"] == 2
    rec = client.get(f"/sessions/{s}/recommendations?variant=matrix").json()
    assert client.post(f"/sessions/{s}/replay", json=payload).json()["history_length"] == 2
    assert client.get(f"/sessions/{s}/recommendations?variant=matrix").json() == rec
    assert client.post(f"/sessions/{s}/reset").json()["history_length"] == 0


def test_api_catalog_and_health():
    health = client.get("/health").json()
    assert health["model_status"] == "toy_untrained"
    items = client.get("/items", params={"query": "Racing"}).json()["items"]
    assert len(items) == 2
    assert all(x["source"] == "fictional_demo" for x in items)


def test_session_default_variant_and_strict_rating_input():
    created = client.post("/sessions", json={"variant": "vector"})
    assert created.status_code == 200
    sid = created.json()["session_id"]
    assert created.json()["default_variant"] == "vector"
    get_default = client.get(f"/sessions/{sid}/recommendations")
    assert get_default.status_code == 200
    assert get_default.json()["variant"] == "vector"
    assert client.post("/sessions", json={"variant": "unsupported"}).status_code == 422
    for bad in [4.5, "5", None]:
        assert client.post(
            f"/sessions/{sid}/ratings",
            json={"item_id": "demo-astro", "rating": bad},
        ).status_code == 422


def test_replay_invalid_item_keeps_existing_history():
    sid = new_session()
    client.post(f"/sessions/{sid}/ratings", json={"item_id": "demo-astro", "rating": 5})
    before = client.get(f"/sessions/{sid}/recommendations?variant=matrix").json()
    invalid = {"events": [{"item_id": "demo-rally", "rating": 4},
                          {"item_id": "missing-item", "rating": 5}]}
    assert client.post(f"/sessions/{sid}/replay", json=invalid).status_code == 404
    assert client.get(f"/sessions/{sid}/recommendations?variant=matrix").json() == before
