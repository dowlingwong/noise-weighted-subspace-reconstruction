"""Exact signed authority and review checks, before recording path construction."""
import json
import re
from pathlib import Path
from experiments.mlst2026.hcp_rnoise_20261019 import protocol as base
from experiments.mlst2026.hcp_rnoise_20261019.runner import verify_kit as base_verify
from experiments.mlst2026.meg_z_20261013.runner import kit_hashes

PACKAGE = Path(__file__).resolve().parent
DECISIONS_SHA = 'b5b4d61af57c4f35cb51492766d4354498a7e31024429033bce3ecea7f02b98e'
ADDENDUM_SHA = '5d6dd370a9cbcf80208e359ac7dbb352bfaea0c4e8e3899829fa5739bdc29b9f'
DELTA_SHA = '548a59fa0169c23c5afaed58d54460cf2f473aa98b43dcdad8810194cb83a657'
BASE_ID = '79dcb96822822df9200b3692b2cb3ed34687c2e485ab2e349811a6f39289b364'


def verify():
    actual = kit_hashes(PACKAGE)
    if actual != json.loads((PACKAGE / 'KIT_SHA256.json').read_text()):
        raise RuntimeError('STOP dev kit identity mismatch')
    inherited = base_verify()
    if inherited['kit_identity'] != BASE_ID:
        raise RuntimeError('STOP inherited HCP identity mismatch')
    if base.sha(PACKAGE / 'CONFIG_DELTA.json') != DELTA_SHA:
        raise RuntimeError('STOP config delta differs')
    return {**actual, 'base': inherited}


def config():
    cfg = base.load()
    if base.sha(PACKAGE / 'CONFIG_DELTA.json') != DELTA_SHA:
        raise RuntimeError('STOP config delta differs')
    cfg['gates']['artifact'].update(json.loads((PACKAGE / 'CONFIG_DELTA.json').read_text())['gates']['artifact'])
    return cfg


def date_guard(date):
    # Pure frozen metadata. No recording directory is constructed or examined.
    unit = base.unit(date)
    if unit['split'] != 'dev' or date not in base.load()['access']['dev_dates']:
        raise PermissionError('STOP confirmation/non-dev date prohibited')
    return unit


def accepted_review(path, required):
    text = Path(path).read_text()
    if not re.search(r'^Verdict:\s*\*{0,2}ACCEPTED(?: WITH CHANGES)?\*{0,2}\s*$', text, re.M):
        raise PermissionError('STOP independent review is not accepted')
    if not all(value in text for value in required):
        raise PermissionError('STOP review does not bind current evidence')


def authorize(dates, ms, review):
    units = [date_guard(date) for date in dates]  # reject every bad date FIRST
    expected = base.load()['access']['dev_dates']
    if len(dates) != 12 or sorted(dates) != expected:
        raise PermissionError('STOP exactly all 12 dev dates required, no duplicates')
    ms = Path(ms)
    sheet = ms / 'reviews/hcp_rnoise_author_decisions_20261019/DECISIONS.md'
    if not sheet.is_file() or base.sha(sheet) != DECISIONS_SHA:
        raise PermissionError('STOP missing, unsigned or changed DECISIONS.md')
    text = sheet.read_text()
    if '**Status: SIGNED ' not in text or 'Approved by: **Dowling Wong**' not in text:
        raise PermissionError('STOP unsigned DECISIONS.md')
    add = ms / 'reviews/hcp_rnoise_design_addendum_20261019'
    if base.sha(add / 'ADDENDUM.md') != ADDENDUM_SHA or base.sha(add / 'CONFIG_DELTA.json') != DELTA_SHA:
        raise PermissionError('STOP addendum differs')
    accepted_review(ms / 'reviews/hcp_rnoise_design_addendum_review_20261019_01/REPORT.md',
                    [ADDENDUM_SHA, DELTA_SHA, DECISIONS_SHA])
    identity = verify()
    review = Path(review).resolve()
    if (review.parent.parent != (ms / 'reviews').resolve()
            or not re.fullmatch(r'hcp_rnoise_dev_kit_review_20261019_0[1-3]', review.parent.name)):
        raise PermissionError('STOP implementation review path outside allowed fresh rounds')
    accepted_review(review, [identity['kit_identity'], ADDENDUM_SHA, DECISIONS_SHA])
    return sorted(units, key=lambda u: u['measurement_date'])
