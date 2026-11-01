import pytest

from commodity_finance_assistance.cli import handle_mode_command


def test_mode_command_displays_current_mode(capsys):
    result = handle_mode_command("/mode", "general")

    assert result == "general"

    output = capsys.readouterr().out
    assert "Current mode: general" in output
    assert "trader" in output
    assert "investor" in output


@pytest.mark.parametrize(
    "command, expected",
    [
        ("/mode trader", "trader"),
        ("/mode investor", "investor"),
        ("/mode general", "general"),
        ("/MODE TRADER", "trader"),
    ],
)
def test_valid_mode_switch(command, expected, capsys):
    result = handle_mode_command(command, "general")

    assert result == expected
    assert f"{expected.upper()}" in capsys.readouterr().out


def test_invalid_mode_preserves_current_mode(capsys):
    result = handle_mode_command("/mode administrator", "trader")

    assert result == "trader"
    assert "Mode error:" in capsys.readouterr().out


def test_invalid_mode_does_not_change_investor_mode(capsys):
    result = handle_mode_command("/mode unknown", "investor")

    assert result == "investor"