import pytest

import m13


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the tests that read the full session when it isn't mounted."""
    if m13.SESSION.exists():
        return
    skip = pytest.mark.skip(reason=f"{m13.SESSION} is not mounted")
    for item in items:
        if "nas" in item.keywords:
            item.add_marker(skip)
