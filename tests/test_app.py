import base64
import io

from PIL import Image, ImageDraw

import app as digit_app


def make_zero_data_url():
    image = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((42, 12, 118, 148), outline="black", width=16)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def test_health_endpoint():
    response = digit_app.app.test_client().get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_missing_image_has_readable_error():
    response = digit_app.app.test_client().post("/api/predict", json={})
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_prediction_contract():
    client = digit_app.app.test_client()
    response = client.post("/api/predict", json={"image": make_zero_data_url()})
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data["prediction"], int)
    assert 0 <= data["prediction"] <= 9
    assert isinstance(data["confidence"], float)
    assert set(data["probabilities"]) == {str(number) for number in range(10)}
    assert all(0.0 <= value <= 100.0 for value in data["probabilities"].values())
    assert abs(sum(data["probabilities"].values()) - 100.0) < 2.0


def test_invalid_base64_has_readable_error():
    response = digit_app.app.test_client().post(
        "/api/predict", json={"image": "data:image/png;base64,not-base64"}
    )
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_invalid_image_bytes_have_readable_error():
    encoded = base64.b64encode(b"not an image").decode("ascii")
    response = digit_app.app.test_client().post(
        "/api/predict", json={"image": f"data:image/png;base64,{encoded}"}
    )
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_blank_canvas_is_rejected():
    image = Image.new("RGB", (120, 120), "black")
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    response = digit_app.app.test_client().post(
        "/api/predict", json={"image": f"data:image/png;base64,{encoded}"}
    )
    assert response.status_code == 400
    assert "digit" in response.get_json()["error"].lower()
