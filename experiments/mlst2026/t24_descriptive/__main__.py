import argparse, json, sys
from . import workflow as w


def main():
    p = argparse.ArgumentParser(description='T24-D: descriptive CRESST post-whitening moments (no tests, no p values)')
    s = p.add_subparsers(dest='command', required=True)
    r = s.add_parser('run'); r.add_argument('--data-root', required=True); r.add_argument('--out', required=True)
    v = s.add_parser('verify'); v.add_argument('--root', required=True); v.add_argument('--data-root')
    a = p.parse_args()
    if a.command == 'run':
        w.run(a.data_root, a.out)
    else:
        print(json.dumps(w.verify(a.root, a.data_root)))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e), file=sys.stderr); sys.exit(2)
