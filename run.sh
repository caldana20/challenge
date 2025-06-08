#!/usr/bin/env bash
# ---------------------------------------------------------------
#  Black Box Challenge – reference implementation wrapper
#  Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>
#  Example: ./run.sh 5 885 1226.68
# ---------------------------------------------------------------

set -euo pipefail

# ---- 1  Argument sanity ----------------------------------------------------
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <trip_duration_days> <miles_traveled> <total_receipts_amount>" >&2
  exit 64               # EX_USAGE
fi

# ---- 2  Dispatch to the Python package ------------------------------------
python - "$@" <<'PY'
import sys
from decimal import Decimal, ROUND_HALF_UP

from acme_legacy_clone import reimburse   # ← your trained model API

days      = int(sys.argv[1])
miles     = float(sys.argv[2])
receipts  = float(sys.argv[3])

amount = reimburse(days, miles, receipts)

# Ensure classic “bankers’ rounding” to two decimals
print(Decimal(amount).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
PY
