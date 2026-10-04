"""Unit-conversion and constant checks."""

import numpy as np
import pytest

from seisgeomech import units as u


def test_foot_is_exact():
    assert u.FT_TO_M == 0.3048


def test_standard_gravity_is_the_si_defined_value():
    assert u.STANDARD_GRAVITY == 9.80665


def test_feet_to_metres_roundtrip():
    ft = np.array([0.0, 181.1024, 12631.2336])
    assert np.allclose(u.feet_to_metres(ft) / u.FT_TO_M, ft)


@pytest.mark.parametrize(
    "dt_us_per_ft, expected_m_s",
    [
        (100.0, 3048.0),   # 100 us/ft -> 10 000 ft/s -> 3048 m/s exactly
        (50.0, 6096.0),
        (200.0, 1524.0),
    ],
)
def test_slowness_to_velocity_exact_cases(dt_us_per_ft, expected_m_s):
    assert u.slowness_us_per_ft_to_velocity_m_s(dt_us_per_ft) == pytest.approx(
        expected_m_s, rel=1e-12
    )


def test_slowness_conversion_is_self_inverse():
    dt = np.array([60.0, 75.0, 140.0])
    vp = u.slowness_us_per_ft_to_velocity_m_s(dt)
    back = u.FT_TO_M / vp / u.US_TO_S
    assert np.allclose(back, dt)


def test_density_conversions():
    assert u.gcc_to_kg_m3(2.65) == pytest.approx(2650.0)
    assert u.kg_m3_to_gcc(2650.0) == pytest.approx(2.65)
