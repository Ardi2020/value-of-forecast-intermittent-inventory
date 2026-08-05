"""Recompute every hash in MANIFEST.md and fail if one does not match.

Round 4 found a stale checksum in the manifest: a result file had been regenerated after
the manifest was written, so the recorded hash pointed at a version that no longer
existed. A hand-maintained manifest will drift again, so this check runs in CI on every
push and the build fails rather than the drift going unnoticed.

    python verify_manifest.py
"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(r'^\|\s*`(realcase_[^`]+)`\s*\|[^|]*\|\s*`([0-9a-f]{64})`\s*\|\s*([\d,]+)\s*\|')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = os.path.join(HERE, 'MANIFEST.md')
    if not os.path.exists(manifest):
        sys.exit('MANIFEST.md is missing')

    recorded = {}
    for line in open(manifest, encoding='utf8'):
        m = ROW.match(line.strip())
        if m:
            recorded[m.group(1)] = (m.group(2), int(m.group(3).replace(',', '')))

    on_disk = {f for f in os.listdir(HERE) if f.startswith('realcase_')}
    problems = []

    for missing in sorted(on_disk - set(recorded)):
        problems.append(f'{missing}: present in the repository but absent from MANIFEST.md')
    for gone in sorted(set(recorded) - on_disk):
        problems.append(f'{gone}: recorded in MANIFEST.md but not present')

    for name in sorted(on_disk & set(recorded)):
        want_hash, want_size = recorded[name]
        path = os.path.join(HERE, name)
        got_hash, got_size = sha256(path), os.path.getsize(path)
        if got_hash != want_hash:
            problems.append(f'{name}: hash {got_hash[:16]}… does not match recorded '
                            f'{want_hash[:16]}…')
        elif got_size != want_size:
            problems.append(f'{name}: size {got_size} does not match recorded {want_size}')

    if problems:
        print('MANIFEST.md does not describe the released files:\n')
        for p in problems:
            print('  -', p)
        sys.exit(1)

    print(f'ok: MANIFEST.md matches all {len(on_disk)} released real-case files '
          f'(full SHA-256 and byte size)')


if __name__ == '__main__':
    main()
