r"""Local receipt workspace: .venv\Scripts\python.exe -m App.web_server --port 8023."""
from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import re
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import time
from urllib.parse import parse_qs, unquote, urlparse

import cv2

from App.ocr_catalog_eval import PASS_SIMILARITY, ROOT
from App.web_pipeline import ReceiptPipeline
from App.receipt_editing import apply_edit, save_receipt, update_receipt

STATIC = Path(__file__).parent / "web"
JOB_ID = re.compile(r"^[a-f0-9]{32}$")
DOWNLOADS = {"receipt.json", "receipt.csv", "ocr_rectified.txt", "ocr_rectified.json", "ocr_header.json",
             "layout.json", "rectification.json", "rectification_map.npz", "quantity_cells.json", "numeric_split.json", "row_split.json", "upload.bin"}


class Job:
    def __init__(self, job_id, filename, directory):
        self.id, self.filename, self.directory = job_id, filename, directory
        self.status, self.progress, self.error = "queued", 0, ""
        self.result, self.artifacts, self.events = None, [], []
        self.created_at = time()
        self.condition = threading.Condition()

    def public(self):
        return dict(id=self.id, filename=self.filename, status=self.status, progress=self.progress,
                    error=self.error, result=self.result, artifacts=self.artifacts, created_at=self.created_at)

    def publish(self, event_type, **payload):
        with self.condition:
            if "progress" in payload:
                self.progress = max(self.progress, int(payload["progress"]))
                payload["progress"] = self.progress
            event = dict(type=event_type, id=len(self.events)+1, **copy.deepcopy(payload))
            self.events.append(event)
            self.condition.notify_all()

    def phase(self, phase, progress, title):
        self.publish("phase", phase=phase, progress=progress, title=title)

    def json_file(self, name, data):
        (self.directory / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def image(self, key, image, title, subtitle="", role="derived"):
        path = self.directory / f"{key}.png"
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Не удалось сохранить промежуточное изображение.")
        path.write_bytes(encoded.tobytes())
        asset = dict(key=key, filename=path.name, url=f"/api/jobs/{self.id}/files/{path.name}",
                     width=image.shape[1], height=image.shape[0], title=title, subtitle=subtitle, role=role)
        with self.condition:
            self.artifacts.append(asset)
        self.publish("artifact", artifact=asset)

    def persist(self):
        self.json_file("state.json", self.public())


class JobStore:
    def __init__(self, root, processor=None, max_upload=20*1024*1024):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_upload = max_upload
        self.jobs, self.lock = {}, threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="receipt")
        self.pipeline = ReceiptPipeline()
        self.processor = processor or self.pipeline.process
        # Completed results survive a server restart. Interrupted work is explicit.
        for state_file in sorted(self.root.glob("*/state.json"), key=lambda p:p.stat().st_mtime)[-100:]:
            if not JOB_ID.fullmatch(state_file.parent.name):
                continue
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                job = Job(state_file.parent.name, state["filename"], state_file.parent)
                for key in ("status", "progress", "error", "result", "artifacts", "created_at"):
                    if key in state:
                        setattr(job, key, state[key])
                if job.status not in {"done", "failed"}:
                    job.status, job.error = "failed", "Обработка прервана перезапуском сервера. Загрузите фото ещё раз."
                if job.status == "done" and job.result and "rows" in job.result:
                    update_receipt(job.result)
                    save_receipt(job.directory, job.result)
                job.publish("complete" if job.status == "done" else "error", result=job.result, title=job.error, progress=job.progress)
                self.jobs[job.id] = job
            except (ValueError, KeyError, OSError):
                continue

    def submit(self, data, filename):
        job_id = uuid.uuid4().hex
        filename = re.sub(r"[\x00-\x1f]", "", filename.replace("\\", "/").split("/")[-1])[:200] or "receipt.jpg"
        directory = self.root / job_id
        directory.mkdir()
        (directory / "upload.bin").write_bytes(data)
        job = Job(job_id, filename, directory)
        job.persist()
        with self.lock:
            self.jobs[job_id] = job
        job.publish("phase", phase="queued", title="Фото загружено. Ожидаем обработку", progress=1)
        self.pool.submit(self._process, job)
        return job

    def _process(self, job):
        job.status = "processing"
        try:
            result = self.processor(job)
            with job.condition:
                job.result, job.status = result, "done"
                job.publish("complete", result=result, progress=100, title="Электронный чек готов")
        except Exception as error:
            (job.directory / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            job.status = "failed"
            job.error = str(error) if isinstance(error, ValueError) else "Не удалось завершить обработку. Попробуйте другое фото."
            job.publish("error", title=job.error, progress=job.progress)
            traceback.print_exc()
        finally:
            job.persist()

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None


class ReceiptHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_server(host="0.0.0.0", port=8023, output_dir=None, processor=None, max_upload=20*1024*1024):
    store = JobStore(output_dir or ROOT / "web_outputs", processor, max_upload)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send_data(self, data, content_type, status=200, attachment=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def send_json(self, data, status=200):
            self.send_data(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self):
            parts = urlparse(self.path)
            path = unquote(parts.path)
            if path in {"/", "/index.html", "/app.js", "/styles.css", "/editing.css"}:
                filename = "index.html" if path == "/" else path.lstrip("/")
                mime = {"html":"text/html", "js":"text/javascript", "css":"text/css"}[filename.split(".")[-1]]
                return self.send_data((STATIC / filename).read_bytes(), mime+"; charset=utf-8")
            if path == "/api/health":
                return self.send_json(dict(ok=True, pass_similarity=PASS_SIMILARITY, max_upload_bytes=store.max_upload))
            if path == "/api/catalog":
                query = parse_qs(parts.query).get("q", [""])[0]
                return self.send_json(dict(items=store.pipeline.search_catalog(query)))
            if path == "/api/jobs":
                with store.lock:
                    jobs = sorted(store.jobs.values(), key=lambda job:job.created_at, reverse=True)[:12]
                return self.send_json([dict(id=job.id, filename=job.filename, status=job.status) for job in jobs])
            route = re.fullmatch(r"/api/jobs/([a-f0-9]{32})(?:/(events|files/[^/]+))?", path)
            if not route:
                return self.send_json(dict(error="Адрес не найден"), 404)
            job = store.get(route[1])
            if job is None:
                return self.send_json(dict(error="Обработка не найдена"), 404)
            suffix = route[2]
            if suffix is None:
                return self.send_json(job.public())
            if suffix == "events":
                return self.events(job, parts.query)
            filename = suffix.removeprefix("files/")
            allowed = DOWNLOADS | {asset["filename"] for asset in job.artifacts}
            if filename not in allowed or not (job.directory / filename).is_file():
                return self.send_json(dict(error="Файл не найден"), 404)
            data = (job.directory / filename).read_bytes()
            attachment = filename if filename in {"receipt.csv", "receipt.json", "ocr_rectified.txt", "rectification_map.npz"} else None
            mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            return self.send_data(data, mime, attachment=attachment)

        def events(self, job, query):
            try:
                after = int(self.headers.get("Last-Event-ID") or parse_qs(query).get("after", [0])[0])
            except (TypeError, ValueError):
                after = 0
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    with job.condition:
                        pending = [event for event in job.events if event["id"] > after]
                        terminal = job.status in {"done", "failed"}
                        if not pending and not terminal:
                            job.condition.wait(timeout=12)
                    for event in pending:
                        payload = json.dumps(event, ensure_ascii=False)
                        self.wfile.write(f'id: {event["id"]}\nevent: pipeline\ndata: {payload}\n\n'.encode())
                        after = event["id"]
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    if terminal and after >= len(job.events):
                        break
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            self.close_connection = True

        def do_POST(self):
            parts = urlparse(self.path)
            edit_route = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/edits", parts.path)
            if parts.path != "/api/jobs" and edit_route is None:
                self.close_connection = True
                return self.send_json(dict(error="Адрес не найден"), 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= store.max_upload:
                self.close_connection = True
                return self.send_json(dict(error=f"Выберите изображение размером до {store.max_upload//1024//1024} МБ."), 413)
            if edit_route and length > 65536:
                self.close_connection = True
                return self.send_json(dict(error="Слишком большой запрос редактирования"), 413)
            data = self.rfile.read(length)
            if len(data) != length:
                return self.send_json(dict(error="Изображение загружено не полностью"), 400)
            if edit_route:
                job = store.get(edit_route[1])
                if job is None:
                    return self.send_json(dict(error="Чек не найден"), 404)
                try:
                    payload = json.loads(data)
                    if not isinstance(payload, dict):
                        raise ValueError("Некорректные данные редактирования.")
                    if not any(key in payload for key in ("row", "expected_total", "finalize")):
                        raise ValueError("Не указано изменение чека.")
                    with job.condition:
                        if job.status != "done" or not job.result:
                            return self.send_json(dict(error="Дождитесь завершения распознавания."), 409)
                        updated = apply_edit(job.result, payload, store.pipeline.catalog_entry)
                        learned = store.pipeline.remember_selection(job.result, updated, payload, job.id)
                        save_receipt(job.directory, updated)
                        job.result = updated
                        job.persist()
                    return self.send_json(dict(result=updated, mapping_saved=bool(learned)))
                except RuntimeError as error:
                    return self.send_json(dict(error=str(error)), 409)
                except (ValueError, TypeError) as error:
                    return self.send_json(dict(error=str(error)), 400)
            filename = parse_qs(parts.query).get("filename", ["receipt.jpg"])[0]
            job = store.submit(data, filename)
            return self.send_json(dict(id=job.id, status=job.status), 202)

    server = ReceiptHTTPServer((host, port), Handler)
    server.store = store
    return server


def main():
    parser = argparse.ArgumentParser(description="Локальный веб-интерфейс чеков Торгсофт")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "web_outputs")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("Используйте проектный .venv: .\\.venv\\Scripts\\python.exe -m App.web_server")
    server = create_server(args.host, args.port, args.output_dir)
    server.store.pipeline.knowledge.initialize()
    print(f"Receipt workspace: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Listening on {args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.store.pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
