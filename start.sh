#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

uvicorn main:app --reload --reload-include="templates/*.html" --reload-include="static/*.css"
