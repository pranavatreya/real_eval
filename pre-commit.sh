#!/bin/bash

# This script fails when any of its commands fail.
set -e

# Which directories to check
TARGETS="central_server evaluation_client"

# Python style checks and linting
black --check --diff $TARGETS || (
  echo ""
  echo ""
  echo "The code formatting check failed. To fix the formatting, run:"
  echo ""
  echo -e "\tblack $TARGETS"
  echo ""
  exit 1
)
mypy --install-types --non-interactive $TARGETS

echo "Done."