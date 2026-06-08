from pathlib import Path


def test_custom_map_app_exposes_map_controls():
    app_dir = Path("custom_map_app")
    html = (app_dir / "index.html").read_text(encoding="utf-8")
    script = (app_dir / "app.js").read_text(encoding="utf-8")
    styles = (app_dir / "styles.css").read_text(encoding="utf-8")

    required_ids = [
        "mapCanvas",
        "robotX",
        "robotY",
        "robotTheta",
        "goalX",
        "goalY",
        "motionModel",
        "humanTableBody",
        "exportJson",
        "importJson",
        "copyEvalCommand",
    ]
    for element_id in required_ids:
        assert f'id="{element_id}"' in html

    assert "speed" in script
    assert "theta_deg" in script
    assert "human_motion_model" in script
    assert "evaluate_custom_scenario.py" in script
    assert "draw({ table: false })" in script
    assert "grid" in styles
