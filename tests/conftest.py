"""Shared test fixtures and early mocking for the topoprm test suite.

Mocks ``swift.rewards`` at import time so that reward modules can be
imported without the full swift installation.
"""

import sys
from types import ModuleType

_swift = ModuleType("swift")
_swift_rewards = ModuleType("swift.rewards")
_swift_rewards_orm = ModuleType("swift.rewards.orm")


class _ORM:
    """Minimal stand-in for ``swift.rewards.ORM``."""
    pass


_swift_rewards.ORM = _ORM
_swift_rewards.orms = {}
_swift_rewards_orm.ORM = _ORM
_swift_rewards_orm.orms = _swift_rewards.orms

sys.modules.setdefault("swift", _swift)
sys.modules.setdefault("swift.rewards", _swift_rewards)
sys.modules.setdefault("swift.rewards.orm", _swift_rewards_orm)
