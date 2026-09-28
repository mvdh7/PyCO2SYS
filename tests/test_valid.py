# %%
import PyCO2SYS as pyco2


def test_valid_zeroes():
    co2s = (
        pyco2.sys(
            dic=0,
            alkalinity=2250,
            tb=0,
        )
        .solve("pH")
        .check_valid("pH")
    )
    assert co2s.v.ph
    co2s = (
        pyco2.sys(
            dic=2100,
            alkalinity=2250,
            tb=0,
        )
        .solve("pH")
        .check_valid("pH")
    )
    assert co2s.v.ph
    co2s = (
        pyco2.sys(
            dic=0,
            alkalinity=2250,
            tb=400,
        )
        .solve("pH")
        .check_valid("pH")
    )
    assert co2s.v.ph
    co2s = (
        pyco2.sys(
            dic=2100,
            alkalinity=2250,
            tb=400,
            pressure=0,
        )
        .solve("pH")
        .check_valid("pH")
    )
    assert co2s.v.ph
    co2s = (
        pyco2.sys(
            dic=2100,
            alkalinity=2250,
            tb=400,
            pressure=1,
        )
        .solve("pH")
        .check_valid("pH")
    )
    assert not co2s.v.ph


# test_valid_zeroes()
