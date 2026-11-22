"""Add missing CDS client dependencies inside wind-env while constraining every existing package."""
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    return {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}


def main():
    job = os.environ.get('SLURM_JOB_ID')
    if not job or Path(sys.prefix).resolve() != ROOT/'wind-env':
        raise ValueError('Use sbatch slurm/run_cerra_download_setup.sh with project wind-env.')
    out = ROOT/'environment'/f'cds_setup_{job}'
    out.mkdir(parents=True, exist_ok=False)
    with (ROOT/'environment/cds_install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = snapshot()
        constraints = out/'constraints.txt'
        constraints.write_text('\n'.join(f'{name}=={version}' for name, version in sorted(before.items()))+'\n')
        config = out/'empty_pip.conf'
        config.write_text('')
        (out/'before.json').write_text(json.dumps(before, indent=2)+'\n')
        # Do not upgrade installed packages to satisfy a new client: fail on conflicts.
        subprocess.run([sys.executable, '-m', 'pip', '--disable-pip-version-check', 'install',
                        '--no-cache-dir', '--constraint', str(constraints), 'cdsapi>=0.7.7,<0.8'], check=True,
                       env=os.environ | {'PIP_CONFIG_FILE': str(config), 'PIP_CACHE_DIR': str(out/'cache'),
                                         'TMPDIR': str(out), 'PYTHONNOUSERSITE': '1'})
        after = snapshot()
        changed = {name: [version, after.get(name)] for name, version in before.items() if after.get(name) != version}
        if changed:
            raise ValueError('Existing versions changed unexpectedly; stop before any scientific work.')
        import cdsapi  # noqa: F401
        (out/'completion.json').write_text(json.dumps(dict(status='passed', existing_versions_unchanged=True,
            added={name: value for name, value in after.items() if name not in before},
            cdsapi_version=importlib.metadata.version('cdsapi')), indent=2)+'\n')
        print('CDS client ready in wind-env; all pre-existing package versions unchanged.')


if __name__ == '__main__':
    main()
