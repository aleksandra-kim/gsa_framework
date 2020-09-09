from gsa_framework.sampling import Directions
import numpy as np
import pytest


def test_directions_dtype():
    directions = Directions()
    assert directions[6].dtype == np.int64


def test_directions_data():
    directions = Directions()
    assert directions.data.shape == (20999, 19)

    assert np.allclose(directions[0], [0, 1])
    assert np.allclose(directions[6], [2, 1, 1, 5, 5, 17])
    assert np.allclose(
        directions[-1],
        [
            127696,
            1,
            1,
            3,
            1,
            19,
            61,
            9,
            157,
            341,
            955,
            589,
            2129,
            5683,
            8553,
            16977,
            48629,
            30845,
            14035,
        ],
    )


def test_directions_index_error():
    directions = Directions()
    with pytest.raises(ValueError):
        directions["a"]
