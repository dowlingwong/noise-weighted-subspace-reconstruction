#!/bin/sh
set -eu

template_ref="2f4c3dc023ecb66d12f653a75554b6269256a825"
template_base="https://raw.githubusercontent.com/JmlrOrg/dmlr-style-file/${template_ref}"

curl -L --fail --silent --show-error \
  "${template_base}/dmlr2e.sty" \
  -o dmlr2e.sty

printf '%s\n' "Fetched official DMLR template at ${template_ref}."

