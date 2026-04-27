from pathlib import Path
import re

import torch


_EPOCH_CHECKPOINT_RE = re.compile(r"epoch_(\d+)\.pt$")
_STEP_CHECKPOINT_RE = re.compile(r"epoch_(\d+)_step_(\d+)\.pt$")
_DRIVE_MOUNT_PREFIX = ("/content/drive/MyDrive", "/content/drive/Shareddrives")
_DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
_DRIVE_SERVICE = None
_DRIVE_SERVICE_UNAVAILABLE = False


def _checkpoint_sort_key(path):
    epoch_match = _EPOCH_CHECKPOINT_RE.match(path.name)
    if epoch_match:
        return int(epoch_match.group(1))

    return -1


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = [path for path in checkpoint_dir.glob("*.pt") if _checkpoint_sort_key(path) >= 0]
    if not checkpoints:
        return None

    return max(checkpoints, key=_checkpoint_sort_key)


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, device):
    return torch.load(Path(path), map_location=device)


def _escape_drive_query_value(value):
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _get_drive_service():
    global _DRIVE_SERVICE
    global _DRIVE_SERVICE_UNAVAILABLE

    if _DRIVE_SERVICE is not None:
        return _DRIVE_SERVICE
    if _DRIVE_SERVICE_UNAVAILABLE:
        return None

    try:
        from google.colab import auth
        from googleapiclient.discovery import build
    except ImportError:
        _DRIVE_SERVICE_UNAVAILABLE = True
        return None

    try:
        auth.authenticate_user()
        _DRIVE_SERVICE = build("drive", "v3")
    except Exception as exc:
        print(f"warning: could not initialize Drive API for permanent delete ({exc})")
        _DRIVE_SERVICE_UNAVAILABLE = True
        return None

    return _DRIVE_SERVICE


def _get_drive_file_id(path):
    path_str = str(Path(path).resolve())
    if path_str.startswith("/content/drive/MyDrive/"):
        relative_parts = Path(path_str).relative_to("/content/drive/MyDrive").parts
        parent_id = "root"
    elif path_str.startswith("/content/drive/Shareddrives/"):
        relative_parts = Path(path_str).relative_to("/content/drive/Shareddrives").parts
        if len(relative_parts) < 2:
            return None
        drive_name, *relative_parts = relative_parts
        service = _get_drive_service()
        if service is None:
            return None
        response = service.drives().list(fields="drives(id, name)").execute()
        shared_drive = next((d for d in response.get("drives", []) if d["name"] == drive_name), None)
        if shared_drive is None:
            return None
        parent_id = shared_drive["id"]
    else:
        return None

    if not relative_parts:
        return None

    service = _get_drive_service()
    if service is None:
        return None

    for part in relative_parts[:-1]:
        query = (
            f"'{parent_id}' in parents and "
            f"name = '{_escape_drive_query_value(part)}' and "
            f"mimeType = '{_DRIVE_FOLDER_MIME}' and trashed = false"
        )
        response = service.files().list(
            q=query,
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        matches = response.get("files", [])
        if not matches:
            return None
        parent_id = matches[0]["id"]

    final_name = _escape_drive_query_value(relative_parts[-1])
    query = f"'{parent_id}' in parents and name = '{final_name}'"
    response = service.files().list(
        q=query,
        fields="files(id, name, trashed)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    matches = response.get("files", [])
    if not matches:
        return None

    return matches[0]["id"]


def _unlink_and_purge_drive(path):
    path = Path(path)
    drive_file_id = None

    if any(str(path).startswith(prefix) for prefix in _DRIVE_MOUNT_PREFIX):
        drive_file_id = _get_drive_file_id(path)

    path.unlink()

    if drive_file_id is None:
        return

    service = _get_drive_service()
    if service is None:
        return

    try:
        service.files().delete(fileId=drive_file_id, supportsAllDrives=True).execute()
    except Exception as exc:
        print(f"warning: checkpoint deleted locally but not permanently removed from Drive ({exc})")


def prune_checkpoints(checkpoint_dir, keep_last_epoch_checkpoints=5, keep_last_step_checkpoints=3):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    epoch_paths = []
    step_paths = []

    for path in checkpoint_dir.glob("*.pt"):
        epoch_match = _EPOCH_CHECKPOINT_RE.match(path.name)
        if epoch_match:
            epoch_paths.append((int(epoch_match.group(1)), path))
            continue

        step_match = _STEP_CHECKPOINT_RE.match(path.name)
        if step_match:
            epoch_num, step_num = step_match.groups()
            step_paths.append((int(epoch_num), int(step_num), path))

    epoch_paths.sort()
    step_paths.sort()

    epoch_paths_to_keep = {
        path for _, path in epoch_paths[-keep_last_epoch_checkpoints:]
    } if keep_last_epoch_checkpoints > 0 else set()

    step_paths_to_keep = {
        path for _, _, path in step_paths[-keep_last_step_checkpoints:]
    } if keep_last_step_checkpoints > 0 else set()

    for _, path in epoch_paths:
        if path not in epoch_paths_to_keep and path.exists():
            _unlink_and_purge_drive(path)

    for _, _, path in step_paths:
        if path not in step_paths_to_keep and path.exists():
            _unlink_and_purge_drive(path)
