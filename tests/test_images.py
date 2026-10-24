import io

import pytest

pytest.importorskip("PIL")
from PIL import Image

from images import MAX_SIDE, MIN_SIDE, prepare_jpeg


def prepared_size(tmp_path, size):
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", size, (200, 120, 40)).save(photo)
    with Image.open(io.BytesIO(prepare_jpeg(photo))) as result:
        return result.format, result.size


@pytest.mark.parametrize("size", [(300, 225), (100, 50), (4000, 3000), (100, 1000)])
def test_prepared_photo_fits_jev_side_limits_as_jpeg(tmp_path, size):
    fmt, (width, height) = prepared_size(tmp_path, size)

    assert fmt == "JPEG"
    assert MIN_SIDE <= width <= MAX_SIDE
    assert MIN_SIDE <= height <= MAX_SIDE
