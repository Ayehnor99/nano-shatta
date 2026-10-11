#!/bin/sh
# Pre-commit wrapper: run precommit_check.py
# Install: cp precommit_check.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
exec python3 "$(dirname "$0")/../../precommit_check.py"
