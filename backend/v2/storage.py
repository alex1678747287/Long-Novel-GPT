"""SQLite-backed persistence for novels + chapters.

Schema is intentionally small: one row per novel, one row per chapter.
The DB file lives under data/ which is volume-mounted into the container,
so it survives both page refreshes and container restarts.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .registry import DATA_DIR


DB_PATH = DATA_DIR / 'long_novel.db'

_lock = threading.Lock()
_initialized = False


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=5000')
    return conn


def _init() -> None:
    global _initialized
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        conn = _connect()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS novels (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    genre TEXT DEFAULT '',
                    target_genre TEXT DEFAULT '',
                    style_tone TEXT DEFAULT '',
                    rewrite_strength TEXT DEFAULT '',
                    split_mode TEXT DEFAULT '',
                    analysis TEXT DEFAULT '',
                    analysis_status TEXT DEFAULT 'idle',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chapters (
                    id TEXT PRIMARY KEY,
                    novel_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    content TEXT DEFAULT '',
                    rewritten TEXT DEFAULT '',
                    rewritten_script TEXT DEFAULT '',
                    script_status TEXT DEFAULT 'idle',
                    overlap REAL,
                    quality_score REAL,
                    quality_grade TEXT DEFAULT '',
                    quality_issues TEXT DEFAULT '',
                    status TEXT DEFAULT 'idle',
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (novel_id) REFERENCES novels(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS chapters_by_novel
                    ON chapters (novel_id, idx);

                CREATE TABLE IF NOT EXISTS rewrite_jobs (
                    id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    novel_id TEXT NOT NULL,
                    chapter_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    phase TEXT NOT NULL DEFAULT 'queued',
                    progress INTEGER NOT NULL DEFAULT 0,
                    model_id TEXT DEFAULT '',
                    prompt_id TEXT DEFAULT '',
                    payload_json TEXT DEFAULT '{}',
                    result_json TEXT DEFAULT '',
                    error TEXT DEFAULT '',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    locked_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    finished_at REAL,
                    FOREIGN KEY (novel_id) REFERENCES novels(id) ON DELETE CASCADE,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS rewrite_jobs_by_novel_batch
                    ON rewrite_jobs (novel_id, batch_id, created_at);

                CREATE INDEX IF NOT EXISTS rewrite_jobs_by_claim
                    ON rewrite_jobs (status, locked_at, created_at);
            """)

            # Migration: add analysis columns to pre-existing novels tables.
            # SQLite errors if the column already exists, hence the try/except.
            for ddl in [
                "ALTER TABLE novels ADD COLUMN genre TEXT DEFAULT ''",
                "ALTER TABLE novels ADD COLUMN target_genre TEXT DEFAULT ''",
                "ALTER TABLE novels ADD COLUMN style_tone TEXT DEFAULT ''",
                "ALTER TABLE novels ADD COLUMN rewrite_strength TEXT DEFAULT ''",
                "ALTER TABLE novels ADD COLUMN analysis TEXT DEFAULT ''",
                "ALTER TABLE novels ADD COLUMN analysis_status TEXT DEFAULT 'idle'",
                # Variant-specific rewrites — lets the 基础洗稿 and 剧本版 results
                # coexist on the same chapter without overwriting each other.
                "ALTER TABLE chapters ADD COLUMN rewritten_script TEXT DEFAULT ''",
                "ALTER TABLE chapters ADD COLUMN script_status TEXT DEFAULT 'idle'",
                "ALTER TABLE chapters ADD COLUMN quality_score REAL",
                "ALTER TABLE chapters ADD COLUMN quality_grade TEXT DEFAULT ''",
                "ALTER TABLE chapters ADD COLUMN quality_issues TEXT DEFAULT ''",
            ]:
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass
            conn.execute(
                """UPDATE chapters
                   SET script_status = 'done'
                   WHERE COALESCE(rewritten_script, '') <> ''
                     AND COALESCE(script_status, 'idle') = 'idle'"""
            )
            conn.commit()
        finally:
            conn.close()
        _initialized = True


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


# ---- Novels ----

def list_novels() -> list[dict[str, Any]]:
    _init()
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT n.id, n.title, n.genre, n.target_genre, n.style_tone,
                      n.rewrite_strength, n.split_mode, n.analysis_status,
                      n.created_at, n.updated_at,
                      COUNT(c.id) AS chapter_count,
                      SUM(CASE WHEN c.status='done' THEN 1 ELSE 0 END) AS done_count
               FROM novels n LEFT JOIN chapters c ON c.novel_id = n.id
               GROUP BY n.id ORDER BY n.updated_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_novel(novel_id: str) -> dict[str, Any] | None:
    """Return novel + its chapters in idx order."""
    _init()
    conn = _connect()
    try:
        novel = conn.execute(
            'SELECT * FROM novels WHERE id = ?', (novel_id,)
        ).fetchone()
        if not novel:
            return None
        chapters = conn.execute(
            'SELECT * FROM chapters WHERE novel_id = ? ORDER BY idx ASC',
            (novel_id,),
        ).fetchall()
        out = dict(novel)
        out['chapters'] = [dict(c) for c in chapters]
        return out
    finally:
        conn.close()


def create_novel(
    title: str,
    chapters: list[dict[str, Any]],
    split_mode: str = '',
    genre: str = '',
    target_genre: str = '',
    style_tone: str = '',
    rewrite_strength: str = '',
) -> dict[str, Any]:
    """Insert a novel + its chapters in one transaction."""
    _init()
    now = time.time()
    novel_id = uuid.uuid4().hex[:12]
    conn = _connect()
    try:
        with conn:  # implicit transaction
            conn.execute(
                'INSERT INTO novels (id, title, genre, target_genre, style_tone, rewrite_strength, split_mode, created_at, updated_at)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (novel_id, title, genre, target_genre, style_tone, rewrite_strength, split_mode, now, now),
            )
            for i, c in enumerate(chapters):
                conn.execute(
                    """INSERT INTO chapters
                       (id, novel_id, idx, title, summary, content,
                        rewritten, rewritten_script, script_status,
                        overlap, status, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        c.get('id') or uuid.uuid4().hex[:12],
                        novel_id,
                        i,
                        c.get('title') or f'第{i+1}章',
                        c.get('summary') or '',
                        c.get('content') or '',
                        c.get('rewritten') or '',
                        c.get('rewritten_script') or '',
                        c.get('script_status') or ('done' if c.get('rewritten_script') else 'idle'),
                        c.get('overlap'),
                        c.get('status') or 'idle',
                        now,
                    ),
                )
    finally:
        conn.close()
    return get_novel(novel_id)


def update_novel(novel_id: str, **fields: Any) -> dict[str, Any] | None:
    """Update novel meta. Supported fields: title, genre, target_genre,
    style_tone, rewrite_strength, split_mode, analysis, analysis_status."""
    allowed = {
        'title', 'genre', 'target_genre', 'style_tone', 'rewrite_strength',
        'split_mode', 'analysis', 'analysis_status',
    }
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return get_novel(novel_id)
    _init()
    updates['updated_at'] = time.time()
    sets = ', '.join(f'{k} = ?' for k in updates)
    conn = _connect()
    try:
        with conn:
            conn.execute(
                f'UPDATE novels SET {sets} WHERE id = ?',
                (*updates.values(), novel_id),
            )
    finally:
        conn.close()
    return get_novel(novel_id)


def delete_novel(novel_id: str) -> None:
    _init()
    conn = _connect()
    try:
        with conn:
            conn.execute('DELETE FROM novels WHERE id = ?', (novel_id,))
    finally:
        conn.close()


def replace_chapters(
    novel_id: str,
    chapters: list[dict[str, Any]],
    split_mode: str = '',
) -> dict[str, Any] | None:
    """Wipe existing chapters and re-insert. Used after re-splitting."""
    _init()
    now = time.time()
    conn = _connect()
    try:
        with conn:
            conn.execute('DELETE FROM chapters WHERE novel_id = ?', (novel_id,))
            for i, c in enumerate(chapters):
                conn.execute(
                    """INSERT INTO chapters
                       (id, novel_id, idx, title, summary, content,
                        rewritten, rewritten_script, script_status,
                        overlap, status, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        c.get('id') or uuid.uuid4().hex[:12],
                        novel_id,
                        i,
                        c.get('title') or f'第{i+1}章',
                        c.get('summary') or '',
                        c.get('content') or '',
                        c.get('rewritten') or '',
                        c.get('rewritten_script') or '',
                        c.get('script_status') or ('done' if c.get('rewritten_script') else 'idle'),
                        c.get('overlap'),
                        c.get('status') or 'idle',
                        now,
                    ),
                )
            conn.execute(
                """UPDATE novels
                   SET split_mode = ?, analysis = '', analysis_status = 'idle',
                       updated_at = ?
                   WHERE id = ?""",
                (split_mode, now, novel_id),
            )
    finally:
        conn.close()
    return get_novel(novel_id)


# ---- Chapters ----

def get_chapter(chapter_id: str) -> dict[str, Any] | None:
    _init()
    conn = _connect()
    try:
        row = conn.execute(
            'SELECT * FROM chapters WHERE id = ?', (chapter_id,)
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


# ---- Backup / restore ----

def export_all() -> dict[str, Any]:
    """Dump every novel + its chapters into a portable JSON blob. The blob
    can be saved as a file and re-imported later (or on a different host)."""
    _init()
    conn = _connect()
    try:
        novels = [dict(r) for r in conn.execute('SELECT * FROM novels').fetchall()]
        chapters = [dict(r) for r in conn.execute('SELECT * FROM chapters').fetchall()]
        return {
            'version': 1,
            'exported_at': time.time(),
            'novels': novels,
            'chapters': chapters,
        }
    finally:
        conn.close()


def import_all(blob: dict[str, Any], merge: bool = True) -> int:
    """Restore novels + chapters from an export blob.

    merge=True: skip novels whose id already exists (default — safest).
    merge=False: wipe all current novels first (DANGER).
    Returns the number of novels actually inserted.
    """
    _init()
    novels = blob.get('novels') or []
    chapters = blob.get('chapters') or []
    if not isinstance(novels, list) or not isinstance(chapters, list):
        raise ValueError('export blob malformed')

    conn = _connect()
    inserted = 0
    try:
        if merge:
            existing = {row[0] for row in conn.execute('SELECT id FROM novels').fetchall()}
            existing_chapters = {
                row['id']: row['novel_id']
                for row in conn.execute('SELECT id, novel_id FROM chapters').fetchall()
            }
            for c in chapters:
                nid = c.get('novel_id')
                cid = c.get('id')
                if nid in existing:
                    continue
                if cid and cid in existing_chapters:
                    raise ValueError(f'chapter id conflict: {cid}')
        with conn:
            if not merge:
                conn.execute('DELETE FROM novels')  # cascades to chapters
            existing = {row[0] for row in conn.execute('SELECT id FROM novels').fetchall()}
            for n in novels:
                if n.get('id') in existing and merge:
                    continue
                conn.execute(
                    """INSERT INTO novels (id, title, genre, target_genre,
                           style_tone, rewrite_strength, split_mode, analysis,
                           analysis_status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        n.get('id'),
                        n.get('title') or '未命名',
                        n.get('genre') or '',
                        n.get('target_genre') or '',
                        n.get('style_tone') or '',
                        n.get('rewrite_strength') or '',
                        n.get('split_mode') or '',
                        n.get('analysis') or '',
                        n.get('analysis_status') or 'idle',
                        n.get('created_at') or time.time(),
                        n.get('updated_at') or time.time(),
                    ),
                )
                inserted += 1
            for c in chapters:
                nid = c.get('novel_id')
                if merge and nid in existing:
                    continue  # skip chapters whose parent novel we didn't insert
                conn.execute(
                    """INSERT INTO chapters
                       (id, novel_id, idx, title, summary, content,
                        rewritten, rewritten_script, script_status,
                        overlap, quality_score, quality_grade, quality_issues,
                        status, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        c.get('id'),
                        c.get('novel_id'),
                        c.get('idx') or 0,
                        c.get('title') or '',
                        c.get('summary') or '',
                        c.get('content') or '',
                        c.get('rewritten') or '',
                        c.get('rewritten_script') or '',
                        c.get('script_status') or ('done' if c.get('rewritten_script') else 'idle'),
                        c.get('overlap'),
                        c.get('quality_score'),
                        c.get('quality_grade') or '',
                        c.get('quality_issues') or '',
                        c.get('status') or 'idle',
                        c.get('updated_at') or time.time(),
                    ),
                )
    finally:
        conn.close()
    return inserted


def update_chapter(chapter_id: str, **fields: Any) -> dict[str, Any] | None:
    """Update fields on a single chapter. Supported fields:
    title, summary, content, rewritten, rewritten_script, script_status,
    overlap, status.
    Bumps the parent novel's updated_at so the sidebar's 'most recent' order
    stays accurate.
    """
    allowed = {
        'title', 'summary', 'content',
        'rewritten', 'rewritten_script', 'script_status',
        'overlap', 'quality_score', 'quality_grade', 'quality_issues', 'status',
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if 'rewritten_script' in updates and 'script_status' not in updates:
        updates['script_status'] = 'done' if updates.get('rewritten_script') else 'idle'
    if not updates:
        return get_chapter(chapter_id)
    _init()
    now = time.time()
    updates['updated_at'] = now
    sets = ', '.join(f'{k} = ?' for k in updates)
    conn = _connect()
    try:
        with conn:
            conn.execute(
                f'UPDATE chapters SET {sets} WHERE id = ?',
                (*updates.values(), chapter_id),
            )
            # Bump parent novel's updated_at
            conn.execute(
                """UPDATE novels SET updated_at = ?
                   WHERE id = (SELECT novel_id FROM chapters WHERE id = ?)""",
                (now, chapter_id),
            )
    finally:
        conn.close()
    return get_chapter(chapter_id)


# ---- Rewrite jobs ----

def _job_row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def create_rewrite_job(
    *,
    novel_id: str,
    chapter_id: str,
    model_id: str = '',
    prompt_id: str = '',
    payload: dict[str, Any] | None = None,
    batch_id: str | None = None,
) -> dict[str, Any]:
    _init()
    now = time.time()
    job_id = uuid.uuid4().hex
    batch = batch_id or uuid.uuid4().hex
    conn = _connect()
    try:
        conn.execute('BEGIN IMMEDIATE')
        existing = conn.execute(
            """SELECT * FROM rewrite_jobs
               WHERE novel_id = ?
                 AND chapter_id = ?
                 AND status IN ('queued', 'running')
               ORDER BY created_at ASC
               LIMIT 1""",
            (novel_id, chapter_id),
        ).fetchone()
        if existing:
            conn.commit()
            return dict(existing)
        conn.execute(
            """INSERT INTO rewrite_jobs
               (id, batch_id, novel_id, chapter_id, status, phase, progress,
                model_id, prompt_id, payload_json, result_json, error,
                retry_count, locked_at, created_at, updated_at, finished_at)
               VALUES (?, ?, ?, ?, 'queued', 'queued', 0, ?, ?, ?, '', '',
                       0, NULL, ?, ?, NULL)""",
            (
                job_id,
                batch,
                novel_id,
                chapter_id,
                model_id or '',
                prompt_id or '',
                json_dumps(payload or {}),
                now,
                now,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_rewrite_job(job_id)


def get_rewrite_job(job_id: str) -> dict[str, Any] | None:
    _init()
    conn = _connect()
    try:
        row = conn.execute(
            'SELECT * FROM rewrite_jobs WHERE id = ?', (job_id,)
        ).fetchone()
        return _job_row_to_dict(row)
    finally:
        conn.close()


def list_rewrite_jobs(
    novel_id: str,
    batch_id: str | None = None,
    active_only: bool = False,
) -> list[dict[str, Any]]:
    _init()
    where = ['novel_id = ?']
    params: list[Any] = [novel_id]
    if batch_id:
        where.append('batch_id = ?')
        params.append(batch_id)
    if active_only:
        where.append("status IN ('queued', 'running')")
    conn = _connect()
    try:
        rows = conn.execute(
            f"""SELECT * FROM rewrite_jobs
                WHERE {' AND '.join(where)}
                ORDER BY created_at ASC""",
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def list_active_rewrite_novel_ids() -> list[str]:
    _init()
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT DISTINCT novel_id FROM rewrite_jobs
               WHERE status IN ('queued', 'running')
               ORDER BY updated_at ASC"""
        ).fetchall()
        return [row['novel_id'] for row in rows if row['novel_id']]
    finally:
        conn.close()


def claim_rewrite_job(worker_id: str = '', lease_seconds: int = 900) -> dict[str, Any] | None:
    _init()
    now = time.time()
    stale_before = now - max(30, lease_seconds)
    conn = _connect()
    try:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute(
            """WITH running_counts AS (
                   SELECT novel_id, COUNT(*) AS running_count
                   FROM rewrite_jobs
                   WHERE status = 'running'
                     AND locked_at IS NOT NULL
                     AND locked_at >= ?
                   GROUP BY novel_id
               )
               SELECT rewrite_jobs.*
               FROM rewrite_jobs
               LEFT JOIN running_counts
                 ON running_counts.novel_id = rewrite_jobs.novel_id
               WHERE rewrite_jobs.status = 'queued'
                  OR (
                       rewrite_jobs.status = 'running'
                       AND (rewrite_jobs.locked_at IS NULL OR rewrite_jobs.locked_at < ?)
                     )
               ORDER BY
                   CASE WHEN rewrite_jobs.status = 'running' THEN 0 ELSE 1 END,
                   COALESCE(running_counts.running_count, 0) ASC,
                   rewrite_jobs.created_at ASC
               LIMIT 1""",
            (stale_before, stale_before),
        ).fetchone()
        if not row:
            conn.commit()
            return None
        conn.execute(
            """UPDATE rewrite_jobs
               SET status = 'running',
                   phase = CASE WHEN phase = 'queued' THEN 'initial' ELSE phase END,
                   progress = CASE WHEN progress < 5 THEN 5 ELSE progress END,
                   locked_at = ?,
                   updated_at = ?,
                   error = ''
               WHERE id = ?""",
            (now, now, row['id']),
        )
        conn.commit()
        return get_rewrite_job(row['id'])
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_rewrite_job(job_id: str, **fields: Any) -> dict[str, Any] | None:
    expected_locked_at = fields.pop('expected_locked_at', None)
    require_status = fields.pop('require_status', None)
    allowed = {
        'status', 'phase', 'progress', 'payload_json', 'result_json', 'error', 'retry_count',
        'locked_at', 'finished_at',
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if 'progress' in updates:
        try:
            updates['progress'] = max(0, min(100, int(updates['progress'])))
        except (TypeError, ValueError):
            updates.pop('progress', None)
    if not updates:
        return get_rewrite_job(job_id)
    now = time.time()
    updates['updated_at'] = now
    if updates.get('status') in {'done', 'error', 'canceled'} and not updates.get('finished_at'):
        updates['finished_at'] = now
    sets = ', '.join(f'{k} = ?' for k in updates)
    _init()
    conn = _connect()
    try:
        conn.execute('BEGIN IMMEDIATE')
        current = conn.execute(
            'SELECT status, locked_at FROM rewrite_jobs WHERE id = ?', (job_id,)
        ).fetchone()
        if not current:
            conn.commit()
            return None
        current_status = current['status']
        if require_status is not None:
            if isinstance(require_status, (list, tuple, set)):
                required_statuses = set(require_status)
            else:
                required_statuses = {str(require_status)}
            if current_status not in required_statuses:
                conn.commit()
                return None
        if expected_locked_at is not None:
            try:
                current_locked_at = float(current['locked_at'])
                expected = float(expected_locked_at)
            except (TypeError, ValueError):
                conn.commit()
                return None
            if abs(current_locked_at - expected) > 0.000001:
                conn.commit()
                return None
        next_status = updates.get('status')
        if current_status in {'done', 'error', 'canceled'} and next_status != current_status:
            conn.commit()
            return get_rewrite_job(job_id)
        conn.execute(
            f'UPDATE rewrite_jobs SET {sets} WHERE id = ?',
            (*updates.values(), job_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_rewrite_job(job_id)


def recover_running_rewrite_jobs(reason: str = 'worker restarted') -> int:
    """Return running rewrite jobs to the queue after a confirmed process restart."""
    _init()
    now = time.time()
    message = f'自动恢复排队：{reason}'
    conn = _connect()
    try:
        conn.execute('BEGIN IMMEDIATE')
        cur = conn.execute(
            """UPDATE rewrite_jobs
               SET status = 'queued',
                   phase = 'retry_wait',
                   progress = CASE WHEN progress >= 100 THEN 95 ELSE progress END,
                   locked_at = NULL,
                   updated_at = ?,
                   error = ?
               WHERE status = 'running'""",
            (now, message),
        )
        conn.commit()
        return int(cur.rowcount or 0)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def finish_rewrite_job_with_chapter(
    job_id: str,
    *,
    expected_locked_at: float | None = None,
    result: dict[str, Any] | None = None,
    status: str = 'done',
    phase: str = 'done',
    error: str = '',
    chapter_update: dict[str, Any] | None = None,
    novel_id: str | None = None,
    chapter_id: str | None = None,
) -> dict[str, Any] | None:
    """Atomically finish a job and optionally persist its chapter rewrite."""
    allowed_chapter_fields = {
        'title', 'summary', 'content',
        'rewritten', 'rewritten_script', 'script_status',
        'overlap', 'quality_score', 'quality_grade', 'quality_issues', 'status',
    }
    chapter_updates = {
        k: v for k, v in (chapter_update or {}).items() if k in allowed_chapter_fields
    }
    if 'rewritten_script' in chapter_updates and 'script_status' not in chapter_updates:
        chapter_updates['script_status'] = 'done' if chapter_updates.get('rewritten_script') else 'idle'
    now = time.time()
    _init()
    conn = _connect()
    try:
        conn.execute('BEGIN IMMEDIATE')
        current = conn.execute(
            'SELECT * FROM rewrite_jobs WHERE id = ?', (job_id,)
        ).fetchone()
        if not current:
            conn.commit()
            return None
        current_status = current['status']
        if expected_locked_at is not None:
            if current_status != 'running':
                conn.commit()
                return None
            try:
                current_locked_at = float(current['locked_at'])
                expected = float(expected_locked_at)
            except (TypeError, ValueError):
                conn.commit()
                return None
            if abs(current_locked_at - expected) > 0.000001:
                conn.commit()
                return None
        if current_status in {'done', 'error', 'canceled'} and status != current_status:
            conn.commit()
            return dict(current)
        target_chapter_id = chapter_id or current['chapter_id']
        if chapter_updates:
            chapter = conn.execute(
                'SELECT id, novel_id FROM chapters WHERE id = ?', (target_chapter_id,)
            ).fetchone()
            if not chapter:
                conn.commit()
                return None
            if novel_id and chapter['novel_id'] != novel_id:
                conn.commit()
                return None
            if current['novel_id'] and chapter['novel_id'] != current['novel_id']:
                conn.commit()
                return None
            chapter_updates['updated_at'] = now
            chapter_sets = ', '.join(f'{k} = ?' for k in chapter_updates)
            conn.execute(
                f'UPDATE chapters SET {chapter_sets} WHERE id = ?',
                (*chapter_updates.values(), target_chapter_id),
            )
            conn.execute(
                """UPDATE novels SET updated_at = ?
                   WHERE id = (SELECT novel_id FROM chapters WHERE id = ?)""",
                (now, target_chapter_id),
            )
        job_updates = {
            'status': status,
            'phase': phase,
            'progress': 100,
            'result_json': json_dumps(result or {}),
            'error': '' if status == 'done' else (error or (result or {}).get('error', '')),
            'updated_at': now,
        }
        if status in {'done', 'error', 'canceled'}:
            job_updates['finished_at'] = now
        job_sets = ', '.join(f'{k} = ?' for k in job_updates)
        conn.execute(
            f'UPDATE rewrite_jobs SET {job_sets} WHERE id = ?',
            (*job_updates.values(), job_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_rewrite_job(job_id)


def cancel_rewrite_job(job_id: str) -> dict[str, Any] | None:
    _init()
    now = time.time()
    conn = _connect()
    try:
        with conn:
            row = conn.execute(
                'SELECT chapter_id FROM rewrite_jobs WHERE id = ? AND status IN (\'queued\', \'running\')',
                (job_id,),
            ).fetchone()
            cursor = conn.execute(
                """UPDATE rewrite_jobs
                   SET status = 'canceled',
                       phase = 'canceled',
                       error = '',
                       updated_at = ?,
                       finished_at = ?
                   WHERE id = ?
                     AND status IN ('queued', 'running')""",
                (now, now, job_id),
            )
            if row and row['chapter_id'] and cursor.rowcount:
                conn.execute(
                    'UPDATE chapters SET status = ?, updated_at = ? WHERE id = ?',
                    ('canceled', now, row['chapter_id']),
                )
    finally:
        conn.close()
    return get_rewrite_job(job_id)


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
