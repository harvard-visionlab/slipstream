"""Allow ``python -m slipstream <command>``."""

import sys

from slipstream.cli import main

sys.exit(main())
