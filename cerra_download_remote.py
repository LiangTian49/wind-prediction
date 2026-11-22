"""Persist/resume CDS remote jobs without cancelling or duplicating active requests."""
from datetime import datetime, timezone
import json
import uuid


def matches(actual, expected):
    def canonical(values):
        return {key: sorted(str(v) for v in (value if isinstance(value, list) else [value]))
                for key, value in values.items()}
    return canonical(actual) == canonical(expected)


def save_state(path, value):
    if path.exists():
        backup = path.with_name(path.name+f'.previous_{uuid.uuid4().hex}')
        backup.write_bytes(path.read_bytes())
    temporary = path.with_name(path.name+f'.tmp_{uuid.uuid4().hex}')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def active_jobs(client):
    page = client.client.get_jobs(limit=100, status=['accepted', 'running'])
    count = 0
    while page is not None:
        for identity in page.request_ids:
            remote = client.client.get_remote(identity)
            if remote.collection_id == 'reanalysis-cerra-single-levels':
                yield remote
            count += 1
            if count > 2000:
                raise ValueError('Too many account jobs to establish a safe remote queue inventory.')
        page = page.next


def resume_or_submit(client, dataset, request, state_path):
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['dataset'] != dataset or not matches(state['request'], request):
            raise ValueError('Saved remote request disagrees with the selected monthly request.')
        remote = client.client.get_remote(state['request_id'])
        if remote.collection_id != dataset or not matches(remote.request, request):
            raise ValueError('Remote job identity/parameters changed.')
        status = remote.status
        if status in ('accepted', 'running', 'successful'):
            print(f'[CDS] Resuming saved request {remote.request_id} ({status})', flush=True)
            return remote
        if status not in ('failed', 'rejected', 'dismissed', 'deleted'):
            raise ValueError(f'Unknown remote status {status}; no new request submitted.')
    # Adopt an active matching job before making any new submission. Other active
    # jobs block submission until the recovery inventory has assigned their months.
    active = list(active_jobs(client))
    selected = [r for r in active if matches(r.request, request)]
    if len(selected) > 1:
        raise ValueError('Multiple active matching remote jobs; review them in CDS before retrying.')
    if selected:
        remote = selected[0]
    elif active:
        raise ValueError('Other CERRA requests are still active. Run the remote recovery inventory first; '
                         'no additional CDS request was submitted.')
    else:
        remote = client.retrieve(dataset, request)  # Client must use wait_until_complete=False.
    save_state(state_path, dict(dataset=dataset, request=request, request_id=remote.request_id,
                               recorded_at_utc=datetime.now(timezone.utc).isoformat()))
    print(f'[CDS] Saved request ID {remote.request_id} before waiting.', flush=True)
    return remote
