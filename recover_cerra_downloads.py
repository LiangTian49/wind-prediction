"""Slurm-only read-only CDS job inventory; recover exact monthly matches, never delete jobs."""
from datetime import datetime, timezone
import json
import os

from cerra_download_remote import matches, save_state
from download_cerra_confirmation import API_URL, DATASET, TASKS, monthly_request
from temporal_folds import project_path


def main():
    job, key = os.environ.get('SLURM_JOB_ID'), os.environ.get('CDSAPI_KEY')
    if not job or not key:
        raise ValueError('Submit the recovery job with CDSAPI_KEY; never run API recovery on login nodes.')
    import cdsapi
    os.environ['CDSAPI_RC'] = str(project_path('environment/cdsapi_unused.rc'))
    client = cdsapi.Client(url=API_URL, key=key, verify=True, quiet=True, debug=False, progress=False,
                           timeout=120, retry_max=5, sleep_max=60, wait_until_complete=False)
    candidates = {i: [] for i in range(24)}
    page = client.client.get_jobs(limit=100, sortby='-created', status=['accepted', 'running', 'successful'])
    seen = 0
    unrelated_active = []
    while page is not None:
        for identity in page.request_ids:
            remote = client.client.get_remote(identity)
            if remote.collection_id != DATASET:
                continue
            request, status = remote.request, remote.status
            found = False
            for i, (year, month) in enumerate(TASKS):
                if matches(request, monthly_request(year, month)):
                    candidates[i].append(dict(request_id=identity, status=status))
                    found = True
                    break
            if not found and status in ('accepted', 'running'):
                unrelated_active.append(identity)
            seen += 1
            if seen > 2000:
                raise ValueError('Too many CERRA jobs; stopped without submitting or deleting requests.')
        page = page.next
    report = []
    for index, (year, month) in enumerate(TASKS):
        target = project_path(f'data/CERRA/raw_{year}/cerra_{year}_{month:02d}.grib')
        receipt = target.with_suffix('.download.json')
        choices = candidates[index]
        active = [r for r in choices if r['status'] in ('accepted', 'running')]
        selected = None
        if len(active) == 1:
            selected = active[0]
        elif not active and choices:
            selected = choices[0]
        if selected and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            save_state(target.with_suffix('.remote.json'), dict(dataset=DATASET,
                request=monthly_request(year, month), request_id=selected['request_id'],
                recorded_at_utc=datetime.now(timezone.utc).isoformat(), recovered_by_slurm_job=job))
        report.append(dict(task=index, year=year, month=month, local_completed=target.exists() and receipt.exists(),
                           matches=choices, selected=selected, duplicate_active_jobs=len(active)>1))
    output = project_path(f'result/cerra_download/recovery_{job}')
    output.mkdir(parents=True, exist_ok=False)
    (output/'recovery.json').write_text(json.dumps(dict(status='inventory_completed', months=report,
        unrelated_active_cerra_jobs=unrelated_active,
        scope='Read-only remote inspection and local ID recovery; no submission, cancellation or field decoding'), indent=2)+'\n')
    for row in report:
        print(row['task'], row['year'], row['month'], 'local_complete=', row['local_completed'],
              'remote=', row['selected'], 'duplicate_active=', row['duplicate_active_jobs'], flush=True)
    if unrelated_active or any(row['duplicate_active_jobs'] for row in report):
        raise ValueError('Account has unmatched or duplicate active jobs; review inventory/CDS page before downloading.')
    print(f'[CDS recovery] Inventory completed: {output}')


if __name__ == '__main__':
    main()
