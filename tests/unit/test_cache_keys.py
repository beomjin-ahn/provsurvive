"""Cache keys: an image's size and every generation setting are part of the key."""

from PIL import Image

from provsurvive.runner import marked_master
from provsurvive.transforms import generative


class _Identity:
    """A binding that embeds nothing, so the cache returns exactly what it stored."""

    name, alg = "identity", "test.identity"

    def config(self):
        return {}

    def embed(self, img, payload):
        return img.convert("RGB")


def test_masters_of_equal_bytes_but_different_size_do_not_collide():
    square = Image.new("RGB", (64, 64), (40, 90, 200))
    wide = Image.new("RGB", (128, 32), (40, 90, 200))
    assert square.tobytes() == wide.tobytes()
    assert marked_master(_Identity(), square, "").size == (64, 64)
    assert marked_master(_Identity(), wide, "").size == (128, 32)


def test_generative_key_covers_size_and_settings(monkeypatch):
    square = Image.new("RGB", (64, 64), (40, 90, 200))
    wide = Image.new("RGB", (128, 32), (40, 90, 200))

    def key(img):
        return generative._cache_key(img, "inpaint", {"area": 0.1}, 7, "a prompt")

    assert key(square) == key(square.copy())
    assert key(square) != key(wide)
    before = key(square)
    monkeypatch.setattr(generative, "INPAINT_STEPS", generative.INPAINT_STEPS + 1)
    assert key(square) != before
