"""custom plugin to fail skipped tests after a given date"""
import pytest
from datetime import date

SKIPUNTIL_MARK = "skipuntil"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        (
            f"{SKIPUNTIL_MARK}(reason, until): marks tests that should be skipped until a given deadline."
            f" They will fail if skipped after that date"
        ),
    )


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        for marker in item.iter_markers(name=SKIPUNTIL_MARK):
            until_date = marker.kwargs.get("until", date.today().isoformat())
            reason = marker.kwargs.get("reason", f"test skipped until {until_date}")
            item.add_marker(pytest.mark.skip(reason=reason))
            break


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()

    # Only report skipped tests once
    if call.when != "teardown":
        return

    mark = item.get_closest_marker(SKIPUNTIL_MARK)
    if report is not None and mark is not None:
        until_date = mark.kwargs.get("until", date.today().isoformat())
        deadline = date.fromisoformat(until_date)
        reason = mark.kwargs.get("reason", "")
        msg = f"Skipped test reached its deadline {until_date}."
        if reason:
            msg += f" {reason}"

        if date.today() >= deadline:
            report.outcome = "failed"
            report.longrepr = msg
