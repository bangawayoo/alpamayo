"""Configure sys.path for agent worktree tests.

When running tests from a git worktree, the editable install resolves to the
main repo's src directory. This conftest prepends the worktree's src directory
so that worktree-local module changes are visible during test runs.
"""

import sys
from pathlib import Path

# Insert worktree src at the front so local changes take precedence over
# the editable install from the main repo.
_worktree_src = str(Path(__file__).parent.parent / "src")
if _worktree_src not in sys.path:
    sys.path.insert(0, _worktree_src)
