#!/bin/sh
set -eu

status=0
for pattern in 'wongdowling' '/Users/' 'github.com/' '\subsubsection*{Acknowledgments}' '\subsubsection*{Author Contributions}'; do
  if grep -R -n -i -F --include='*.tex' --include='*.bib' "${pattern}" .; then
    printf '%s\n' "Potential identity leak matched: ${pattern}" >&2
    status=1
  fi
done

if [ "${status}" -ne 0 ]; then
  exit "${status}"
fi

printf '%s\n' 'No configured identity-leak pattern was found.'
