#!/bin/sh
set -eu

template_ref="7bf90efe3a0debbba703c05c43f3ff7e4d4a2992"
template_base="https://raw.githubusercontent.com/JmlrOrg/tmlr-style-file/${template_ref}"

for template_file in tmlr.sty tmlr.bst fancyhdr.sty; do
  curl -L --fail --silent --show-error \
    "${template_base}/${template_file}" \
    -o "${template_file}"
done

printf '%s\n' "Fetched official TMLR template at ${template_ref}."

