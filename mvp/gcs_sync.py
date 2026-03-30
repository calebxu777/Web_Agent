from __future__ import annotations

import json
from pathlib import Path


class MVPGCSSyncClient:
    def __init__(self, bucket_name: str, project_id: str = ""):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self._client = None
        self._bucket = None

    def _get_bucket(self):
        if self._bucket is not None:
            return self._bucket

        from google.cloud import storage

        client_kwargs = {"project": self.project_id} if self.project_id else {}
        self._client = storage.Client(**client_kwargs)
        self._bucket = self._client.bucket(self.bucket_name)
        return self._bucket

    def upload_file(
        self,
        local_path: str | Path,
        blob_name: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        path = Path(local_path)
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(path), content_type=content_type)
        return f"gs://{self.bucket_name}/{blob_name}"

    def append_jsonl_record(self, blob_name: str, record: dict[str, object]) -> str:
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        existing = ""
        if blob.exists():
            existing = blob.download_as_text()

        payload = existing
        if payload and not payload.endswith("\n"):
            payload += "\n"
        payload += json.dumps(record, ensure_ascii=True) + "\n"
        blob.upload_from_string(payload, content_type="application/x-ndjson")
        return f"gs://{self.bucket_name}/{blob_name}"
