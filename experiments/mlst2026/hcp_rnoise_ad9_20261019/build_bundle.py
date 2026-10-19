"""Build the prospective freeze using committed metadata/JSON; exclusive new MS files."""
import argparse
import copy
import json
from pathlib import Path
from .power import BASELINE, FREEZE, DECISIONS, from_committed, git, sha
from .protocol import PACKAGE
from experiments.mlst2026.meg_z_20261013.runner import kit_hashes

DESIGN = Path('reviews/hcp_rnoise_design_20261019')
AD2 = Path('reviews/hcp_rnoise_design_addendum_20261019')


def encoded(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def new(path, raw):
    with path.open('xb') as f:
        f.write(raw)


def make_config(base, patch, split, disclosure):
    cfg = copy.deepcopy(base)
    cfg['gates']['artifact'].update(patch['gates']['artifact'])
    cfg['access']['confirmation_analysis_dates'] = split['confirmation_dates']
    cfg['access']['unused_confirmation_dates'] = []
    # This bundle does not authorize any recording access.
    cfg['status'] = 'FROZEN_AD8a_AD9_AWAITING_AUTHOR_REGISTRATION_AND_AUTHORIZATION'
    cfg['frozen'] = True
    cfg['n_min'] = 18
    cfg['inference'].update(resamples=50, family_size=34)
    cfg['gates']['E1'].update(confirmation_target=60, confirmation_n_min=18, power_max_n=60,
        power_rule='AD8a/AD9 four reference sign cells; strict family34 t half-width < half abs prediction; nine-date R50 SD one-sided80% upper limit; n=8..60; n_min18; no further remedy; below18 passing confirmation dates STOP/no claim')
    cfg['confirmation_units'] = [{'date': r['measurement_date'], 'subject_id': r['subject_id']}
        for r in split['units'] if r['split'] == 'confirmation']
    cfg['power_required_cells'] = ['A/0/reference', 'A/1/reference', 'B/0/reference', 'B/1/reference']
    cfg['AD8a_AD9_disclosure_verbatim'] = disclosure
    cfg['reporting_rule'] = 'Report every outcome, exclusion, unavailable cell, diagnostic and STOP; never replace exclusions or alter family34.'
    cfg['prediction_order'] = 'Per eligible passing date: qualify and construct unshrunk Sref; compute all four fixed and two adaptive rows; publish exact per-date pre_fit prediction bytes to MS origin/main in a commit strictly after the registration tag; before_fit must pass before any learning fit for that date.'
    cfg['confirmation_seed_index'] = '12 + index in sorted complete60 confirmation list; indices12..71; no cohort selection or analysis-order rule'
    cfg['supersession'] = ['AD2 median-relative amplitude screen only',
        'AD8a excludes theta0 falsification cells from power requirement only; all15 reference cells remain in family34',
        'AD9 supersedes AD3 cohort/n_min rule and no-expansion clause; all60 dates, R50, n8..60, n_min18',
        'F4 STOP7a78a8f remains binding for the earlier12-date design; correction addendum supersedes historical access wording']
    return cfg


def protocol_text(cfg, amendment):
    units = '\n'.join(f"| {r['date']} | {r['subject_id']} |" for r in cfg['confirmation_units'])
    disclosure = amendment.split('## Disclosure (verbatim in protocol and paper, including the access-precision clause)\n\n')[1].split('\n## Correction addendum')[0].strip()
    addendum = amendment[amendment.index('## Correction addendum'):].strip()
    return f'''# HCP Rnoise prospective frozen protocol — AD8a/AD9

Frozen for author registration; unregistered and unauthorized for confirmation access.
Authority: signed DECISIONS.md and AD8_POWER_CELLS.md, as superseded prospectively by
AD8a_AD9_AMENDMENT.md at MS e58f315. Accepted DESIGN.md and CONFIG.json and the accepted
AD2 addendum remain immutable source records; their superseded clauses are not operative.
The exact expanded configuration is protocol_frozen.json, encoded with sorted keys,
two-space indentation, finite JSON numbers and one final newline. No time-dependent fields.

## Units, gates, preprocessing and exclusions

Use **all 60** eligible confirmation dates in the committed SPLIT.json, one preselected
lowest-ID scan per date. The list below names design metadata only. There is no selection,
ordering or replacement rule. The dates span 2013–2015; the earlier first12/all2013 cohort
and prohibition on adding48 dates are superseded by AD9. All60 dates must receive a
reported qualification disposition. Exclusions are never replaced. Independent-day analysis
units do not establish independent hardware states. Below **n_min=18** gate-passing dates:
**STOP, no confirmatory claim**, even if individual results appear favorable.

Keep C12, T6, d72, k6; the pinned 12-channel vertex geometry and 1e-6 location tolerance,
device frame1, no reference-channel regression, artifact repair or channel-drop retest.
Missing/incompatible channels, processed history or unexpected shortening exclude the date.
G1 requires SPD unshrunk reference covariance, spatial condition<1e6 and full condition<1e10.
G2 requires all six values within inclusive [.8,1.25]: tr(Sref^-1 Strain)/d,
m_eval/m_train, own-Sref bottom50/middle40/top10 eval/train eigen-group ratios, and
physical tr(Seval)/tr(Strain). Role covariances use their own means and divisor n.
No scale correction, isotropy/spread gate or G3. Cal/train and train/ref group ratios,
generalized eigenvalue extremes, MP edges, scan-third transport, G4 and G5 are descriptive.

AD2 as amended: screen all selected native samples before filtering over the full scan,
including future guards. Amplitude abs(raw[c,t]-median(raw[c,:])) >=5e-10 T excludes;
raw adjacent step >=5e-11 T or any raw nonfinite excludes. Even-length median is the
mean of middle order statistics. For nonfinite channels, median/amplitude count are
unavailable; retain raw nonfinite and finite-pair step counts, report all channel counts
and reasons. A partial amplitude total must not appear complete. Median subtraction is
for screening only; original values enter the pinned FIR. No time mask, repair or retest;
the screen has no calibrated false-exclusion claim.

MNE native FIR1–40 Hz, zero phase/firwin, 6715 taps, f64le SHA256
c173d8ddd85b7252428e6d88deefa007c4a501af62155625c33c23584d18abca.
Native2034.5101318359375 Hz; slice phase0/every20 to101.72550659179687 Hz. Two-second
edges (204 samples), ten-second full blocks (1017), discard first407 samples of every
block, including block0, and incomplete terminal block. Assert FIR support fits margins.
Role cycle ref/train/cal/eval/train; fixed grids never restart. Ref stride1, others strideT.
Windows are time-major c+C*t. No shared FIR support is not proof of independence.

Budgets and geometry remain fixed: ref stride-T count>=8d=576, train supports
1.1*maxN=976.8, calibration>=1.1*512=563.2 and evaluation>=1.1*500=550, with the
required role blocks. Standard complete-scan counts are ref3630/train1111/cal606/eval606
and ref/train/cal/eval blocks6/11/6/6. Require all fixed budgets; no post-outcome clipping
to rescue a shortened unit. A grid {{56,111,222,444,888}}, B grid {{28,56,111,222,444}};
theta0 uses A grid. Fixed training ceiling1010, m256/m512; no larger dimension or m1024.

## Signal, predictions and immutable order

Keep the fixed homogeneous sphere origin/radius.09 m, Sarvas point magnetometers,
dipoles (-.03,0,.04)/(0,1,0) and (.03,.015,.04)/(1,0,0), normalized spatial maps with
fixed sign, no orthogonalization; reject rank deficiency/abs cosine>.95, no alternate
recipe. Temporal filtered bump/derivative bases, centers30/51 ms, width6 ms, native
[-4,4] s grid and sample offsets0,20,...100, QR/source2 projection remain unchanged.
Independent Gaussian latents, weights(1,1), temporal variances(1,.5,.25), kc2/kt3.
One per-date theta0 scale makes tr(K)=206.604; hold it fixed at other theta.

For each passing date, build unshrunk divisor-n Sref on centered ref stride1 windows.
Compute K=Sref^-1/2 Rs Sref^-1/2, M=K+I, best-found product optimum, Delta_k,
a_best, cU/cF, B=cF-cU, point N* and certificate interval, with flip-flop and
metric-corrected Bdagger=lambda_min(SigmaK^1/2 Sref^-1 SigmaK^1/2)*Bcert(KG).
Four fixed theta rows {{0,.12,.25,.40}} remain; adaptive A/B use the unchanged
population root search bracket[.0001,1.2], xtol1e-6, six search starts, final
HOSVD+theta0+62 random starts. Retain all restart/gradient/Hessian diagnostics.
Agreement of64 starts is not global certification. Unavailable roots/certificates
are reported unavailable; no recipe search. Out-of-grid crossings stay out of grid;
no PZ2 claim there. Fixed rows remain regardless of adaptive availability.

Commit and push each date's exact pre_fit predictions on MS origin/main strictly
after the registration tag and **before any learning fit for that date**. The new
array API enforces registration, cohort membership, prediction commit ancestry,
published prediction bytes and date/pre_fit identity. Dev predictions are historical
evidence, never a substitute for future per-confirmation-date predictions.

## Arms, resampling, estimands and decisions

All six arms: reference, shrunk-full m256/m512, Kronecker-MLE m256/m512, isotropic.
Fit reference in Sref; fit estimated metrics in measurement coordinates; score all
in reference metric. Primary clean_ref class contrast D=product minus unrestricted;
negative favors product. Learner tr(P(K+Ntilde)), clean risk tr(K)-tr(P(K-Ntilde)).
Euclidean clean risk and fabricated source-amplitude error are descriptive; no brain,
evoked response, D8, real source-localization or dimensional-scaling claim.

**R50** replaces the old20-resample confirmation convention. Uniform permutations
of whole equal-length active role blocks without replacement, independent cyclic
rotation of fixed-grid windows in each block, concatenation and truncation; a wrap
uses two observed segments, never creates a new boundary window. All evaluation
windows fixed; fresh independent signal latents and role orders each replicate.
Seeds retain PCG64DXSM/SeedSequence and separate prediction, learning and timing
namespaces. Confirmation date indices are12+index in the complete sorted60 metadata
list (12..71), disjoint from dev0..11; this is deterministic seed bookkeeping only.
Within-date blocks/windows/replicates never count as independent dates.

Analyze means over50 resamples per date. Across all passing dates, simultaneous
two-sided Bonferroni Student-t intervals at alpha.05, df=n-1, **family34**:15 reference
cells +4 paired PZ4 interactions +15 isotropic cells. Do not drop unavailable or
theta0 cells from the family. Positive if lower bound>0, negative if upper bound<0,
otherwise undecided; zero-SE cells stay undecided. Incomplete, duplicate, mixed-cell
or nonfinite ledgers cannot support a claim. Report missing outcomes explicitly.

| Estimand | Decision rule |
|---|---|
| PZ1 | Bdagger zero at theta0 and nondecreasing over all four fixed theta rows in every included date, tolerance1e-9. |
| PZ2 | Reference contrast decided negative below and positive above an in-range N* certificate interval; below/above classification requires the relation in every included date. Unavailable/out-of-grid means no claim. |
| PZ3 | All five theta0 reference cells have no decided positive contrast (falsification, zero population cost). |
| PZ4 | Per date average estimated-minus-reference D over the ten A/B cells, paired on the same replicates; both estimators decided negative at m256, m512 point estimate closer to zero. Four interactions retain family34. |
| PZ5 | All15 isotropic cells have no decided negative class contrast. |

## Prospective power and preserved STOP history

POWER_AD9.json recomputes only committed pilot JSON: nine dev dates, R50 SDs,
one-sided80% multiplier sqrt(8/chi2.ppf(.2,8))=1.3196830946. Scan n8..60; every
AD8a-required cell must satisfy strict t_(1-.05/(2*34),n-1)*SD_upper80/sqrt(n)
<.5*abs(mean(a_best+B/N)). Required cells A/0(N56), A/1(N111), B/0(N28), B/1(N56).
All15 reference cells remain fitted/reported/in family34. Theta0 target population
cost0 makes strict half-width<0 infeasible and is excluded from power only.

Joint **n_min18**, binding A/1: n17 half-width.9260822828 fails target.9017595430;
n18 half-width.8895492451 passes by **1.35%**. At60 its half-width is reported in
POWER_AD9.json along with every cell at18/60 and all n8..60. These are projected
precision criteria, not guaranteed test power. No new fit, resample, remedy or
confirmation statistic enters this calculation. It assumes nine-date pilot SDs
transport across2013–2015; only one pilot date is2013.

The binding F4 STOP7a78a8f and later AD8 STOP stay untouched and binding for the
12-date design: A/1 fails at12 even under AD8a. AD8a is a required-cell rule
correction9→4 (R50 projected46→18); AD9 is a prospective enlargement12→60 and
R50 confirmation rule. Neither retroactively establishes the old n_min. POWER.json,
POWER_R20.json, POWER_AD8.json, all F0–F3 records and both STOP reports remain exact.
REPORT.md remains the frozen STOP report; the current hand-back is REPORT_AD9.md.

## Mandatory disclosure (verbatim)

{disclosure}

{addendum}

## Descriptives, limitations and reporting

Report every outcome, exclusion/cause, unavailable prediction or fit, STOP, interval,
optimizer/Hessian/restart flag and deviation. A miss is evidence about applicability,
never permission to tune thresholds, laws, grids, dimensions, recipe or prediction.
Scale c!=1, reference tilt, serial dependence changing IID S4.3 coefficients,
non-Gaussianity/scan-third mixtures, point-coil approximation, shared hardware and
calendar transport remain limitations. AR lags11–19 have no acceptance threshold;
the12-shape AR pool retains all three E1-excluded dev dates. D3-S is descriptive,
never a gate or adjusted prediction; none was run in F0–F4 and no simulation is
authorized here. Do not borrow R1's1.013–1.14 range or substitute an effective N.
Keep all dev pilot readings labelled DEV PILOT, not confirmatory.

## Registration and access barrier

CONFIRMATION_RUN_CLI.md specifies the future launcher contract. The included CLI
performs registration preflight only. There is no recording launcher in this kit.
The guard requires registration with OSF URL, exact git tag/commit, checksum bundle,
nonempty matching .ots and signed author authorization published on origin/main,
plus exact local and inherited identities. The author must mature/verify the OTS
attestation separately. No tag, stamp, registration or confirmation access is
authorized by this freeze. The author guide applies with the explicit corrections
in CONFIRMATION_RUN_CLI.md; confirmation requires separate author direction.

## Complete60 confirmation units (SPLIT metadata only)

| Date | Subject ID |
|---|---|
{units}
'''


CLI = '''# Confirmation run CLI contract — AD9

Implemented metadata-only preflight, from EXP with meg_env and PYTHONPATH=src:.:
```
python -m experiments.mlst2026.hcp_rnoise_ad9_20261019.guard \\
  --ms /ceph/dwong/arXiv-v1-polished-extended-preprint \\
  --registration-record reviews/hcp_rnoise_registration_YYYYMMDD/REGISTRATION.md
```
Refuses with exit2 unless origin/main contains complete registration and signed
authorization evidence. Before invoking, the future separately authorized operator
must fetch origin main and the named tag (never create/move a tag here). OSF URL
must be https://osf.io/<registration>/; tag hcp-rnoise-registration-YYYYMMDD must
resolve to the recorded commit and be ancestor of origin/main. Both tagged and
current main checksum bundle bytes must agree; every checksum entry must match
current main, including all required bundle files. The registered protocol must
equal the local kit protocol; local/new/inherited kit identities must match.
Tagged OTS must exist; current receipt hash must match registration and signed
authorization, permitting timestamp upgrades. No network OSF or cryptographic
Bitcoin attestation check is implemented; author checks these before authorizing.

REGISTRATION.md uses exact fields:
```
OSF URL: https://osf.io/<registration>/
Git tag: hcp-rnoise-registration-YYYYMMDD
Tagged commit: <full commit hash>
Bundle SHA-256: <REGISTRATION_SHA256SUMS digest>
OTS SHA-256: <current receipt digest>
```
Authorization is a separate CONFIRMATION_AUTHORIZATION.md alongside it; use the
unsigned template in this bundle. Placeholder fields never satisfy the guard.

Future recording launcher (SPECIFICATION ONLY; absent and disabled here):
`confirmation-run --ms MS --registration-record RECORD --registered-bundle SHA
--output NEW_DIRECTORY`. It must fetch then preflight **before any recording
listing, path construction, open, hash, header or decode**, atomically refuse an
existing/started output, persist a once-only start/STOP ledger, and process only
all60 named SPLIT units without replacements. All qualification outcomes must be
reported; call protocol.cohort with all60 booleans before inference. Below18 pass:
STOP/no confirmatory claim. No automatic restart after decode or fit failure.

Per passing date: unchanged AD2/preprocessing/gates; compute all fixed/adaptive
predictions with inherited prediction.predict_unit; publish exact date/pre_fit
JSON on MS origin/main strictly after the registration tag; use guard.before_fit
or guard.fit_confirmation before each learning kernel. The array-fit API requires
date_index12+sorted60-index and replicate indices0..49. Qualification, prediction
and learning must use the same Sref and immutable fixed geometry/grids. Use R50 for
every arm and date; require complete included-date x50 ledgers with readings.py.
Use the protocol's PZ1–PZ5 paired estimands/family34; report all outcomes and flags.
The frozen CLI spec does not authorize building or launching the recording runner.

Author guide corrections for AD9 (the guide itself remains untouched): read
REPORT_AD9.md, preserve REPORT.md as the historical STOP, upload POWER_AD9.json
(POWER.json remains historical), and use the protocol's verbatim access-precision
disclosure in OSF/paper. Never claim recordings were never opened/hashed: the
design-stage scoping access is disclosed. Timestamp/check the repository-relative
REGISTRATION_SHA256SUMS from the MS root; tag/register/mature/verify/sign only as
author actions, then obtain a separate confirmation-run prompt. No such action
has been performed by the freeze agent.
'''


AUTH = '''# UNSIGNED — confirmation authorization template

Copy into reviews/hcp_rnoise_registration_YYYYMMDD/CONFIRMATION_AUTHORIZATION.md.
Fill only after OSF registration, tag publication and OTS maturation/verification.
No placeholder is permission. Publish registration and this signed record to MS
origin/main before requesting a separate confirmation-run launcher/prompt.

OSF URL: <https://osf.io/registration/>
Git tag: <hcp-rnoise-registration-YYYYMMDD>
OTS SHA-256: <current REGISTRATION_SHA256SUMS.ots digest>
Author signature: UNSIGNED

When signing use `Author signature: Dowling Wong; YYYY-MM-DD HH:MM timezone`.
I authorize exactly one confirmation run of the registered AD8a/AD9 bundle on
all60 fixed dates, R50, n_min18, with no replacement; below18 passing units STOP
with no confirmatory claim. Exact per-date predictions must be committed/pushed
after the registered tag before any learning fit. Report every outcome and flag.
'''


def build(ms):
    ms = Path(ms)
    # Only tracked metadata and committed JSON are read; never data_root.
    paths = git(ms, 'ls-tree', '-r', '--name-only', BASELINE, 'reviews').decode().splitlines()
    preserved = {}
    for path in paths:
        if not path.startswith('reviews/hcp_rnoise_'):
            continue
        raw = git(ms, 'show', f'{BASELINE}:{path}')
        if raw != (ms / path).read_bytes():
            raise ValueError('STOP earlier record changed: ' + path)
        preserved[path] = sha(raw)
    power = from_committed(ms)
    if power['status'] != 'PASS' or power['n_min'] != 18:
        raise ValueError('STOP F4 failed prospective AD9 design; do not freeze')
    amendment = (ms / DECISIONS / 'AD8a_AD9_AMENDMENT.md').read_text()
    disclosure = amendment.split('## Disclosure (verbatim in protocol and paper, including the access-precision clause)\n\n')[1].split('\n## Correction addendum')[0].strip()
    cfg = make_config(json.loads((ms / DESIGN / 'CONFIG.json').read_bytes()),
        json.loads((ms / AD2 / 'CONFIG_DELTA.json').read_bytes()),
        json.loads((ms / DESIGN / 'SPLIT.json').read_bytes()), disclosure)
    rawcfg = encoded(cfg)
    new(PACKAGE / 'protocol_frozen.json', rawcfg)
    outputs = {'POWER_AD9.json': encoded(power), 'protocol_frozen.json': rawcfg,
        'PROTOCOL_FROZEN.md': protocol_text(cfg, amendment).encode(),
        'CONFIRMATION_RUN_CLI.md': CLI.encode(), 'CONFIRMATION_AUTHORIZATION_TEMPLATE.md': AUTH.encode(),
        'PRESERVED_RECORDS_AD9.json': encoded({'baseline': BASELINE, 'files': preserved,
            'scope': 'Every preexisting tracked HCP Rnoise review record; exact committed bytes retained',
            'REPORT_filename_resolution': 'REPORT.md is frozen STOP; continuation report is REPORT_AD9.md'})}
    # KIT_SHA256 includes this builder and tests. Update only this new candidate's manifest.
    kit = kit_hashes(PACKAGE)
    new(PACKAGE / 'KIT_SHA256.json', encoded(kit))
    outputs['FREEZE_KIT_SHA256.json'] = encoded(kit)
    for name, raw in outputs.items():
        new(ms / FREEZE / name, raw)
    return outputs


def checksums(ms):
    ms = Path(ms)
    names = ['PROTOCOL_FROZEN.md', 'protocol_frozen.json', 'CONFIRMATION_RUN_CLI.md',
             'FREEZE_KIT_SHA256.json', 'DEV_PREDICTIONS.json', 'PILOT_SUMMARY.json',
             'PILOT_R50_REF_SUMMARY.json', 'POWER_AD9.json', 'CONFIRMATION_AUTHORIZATION_TEMPLATE.md',
             'PRESERVED_RECORDS_AD9.json']
    paths = [FREEZE / n for n in names]
    paths += [DESIGN / n for n in ('SPLIT.json', 'DESIGN.md', 'CONFIG.json', 'GEOMETRY.json', 'BUDGETS.json')]
    paths += [DECISIONS / n for n in ('DECISIONS.md', 'AD8_POWER_CELLS.md', 'AD8a_AD9_AMENDMENT.md')]
    paths += [AD2 / n for n in ('ADDENDUM.md', 'CONFIG_DELTA.json')]
    return ''.join(f'{sha((ms / p).read_bytes())}  {p}\n' for p in sorted(paths)).encode()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ms', required=True)
    args = parser.parse_args()
    build(args.ms)
    new(Path(args.ms) / FREEZE / 'REGISTRATION_SHA256SUMS', checksums(args.ms))
