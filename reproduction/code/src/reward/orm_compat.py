"""Optional Swift registration; TRL consumes plain callable reward objects.

Set TOPO_REWARD_BACKEND=standalone for the isolated TRL experiment environment.
This changes only the registry/base interface, never reward computation.
"""
import os
if os.environ.get("TOPO_REWARD_BACKEND") == "standalone":
    class ORM:
        pass
    orms = {}
else:
    from swift.rewards import ORM, orms
