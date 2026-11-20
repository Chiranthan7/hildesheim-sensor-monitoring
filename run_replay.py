"""Component 8: historical replay with dry-run notifications.  usage: python run_replay.py --foundation <dir> --out <dir>"""
import argparse, json, time
from hsm import replay

def main(fd, out, start='2026-08-01'):
    t0 = time.time(); S, I, N, thr = replay.run(fd, out, start); res = replay.evaluate(fd, out, S, I, N, start); replay.demonstrations(fd, out, S)
    print(json.dumps(res, indent=1, default=str)); print('done', round(time.time() - t0), 's')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--foundation', required=True); ap.add_argument('--out', required=True); ap.add_argument('--start', default='2026-08-01')
    a = ap.parse_args(); main(a.foundation, a.out, a.start)
