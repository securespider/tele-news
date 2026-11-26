"""SQLite persistence layer, multi-user RBAC storage, FTS5 search, and article lifecycle."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

from security import harden_file_permissions

_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_VALID_ROLES = {"member", "admin"}
_VALID_STATUSES = {"active", "revoked", "pending"}
_VALID_LIBRARY_STATUSES = {"to_read", "reading", "read", "archived"}


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def parse_iso_utc(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp string into a timezone-aware UTC datetime."""
    if not raw:
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def clean_feed_url(url: str) -> str:
    """Normalize feed URL scheme/host casing and strip fragments without breaking path slashes."""
    stripped = url.strip()
    if not stripped:
        return ""
    parts = urlsplit(stripped)
    if not parts.scheme or not parts.netloc:
        return stripped
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def _format_authors_text(authors_json: str | None) -> str:
    if not authors_json:
        return ""
    try:
        loaded = json.loads(authors_json)
        if isinstance(loaded, list):
            return ", ".join(str(a) for a in loaded if str(a).strip())
    except (TypeError, ValueError):
        pass
    return str(authors_json)


def _format_topics_text(topics_json: str | None) -> str:
    if not topics_json:
        return ""
    try:
        loaded = json.loads(topics_json)
        if isinstance(loaded, list):
            return ", ".join(str(t) for t in loaded if str(t).strip())
    except (TypeError, ValueError):
        pass
    return str(topics_json)


class Database:
    """Thread-safe short-lived-connection SQLite wrapper in WAL mode."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_available = True

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create tables, apply additive column migrations, initialize FTS5 & WAL mode."""
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feeds (
                    url TEXT PRIMARY KEY,
                    title TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    etag TEXT,
                    last_modified TEXT,
                    last_checked_at TEXT,
                    last_error TEXT,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    added_at TEXT NOT NULL
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    comments_url TEXT,
                    title TEXT NOT NULL,
                    source_feed TEXT NOT NULL,
                    published_at TEXT,
                    rss_snippet TEXT,
                    full_text TEXT,
                    hn_comments TEXT,
                    llm_score REAL,
                    relevance_score REAL,
                    relevance_reason TEXT,
                    summary_json TEXT,
                    topics_json TEXT,
                    stage1_attempts INTEGER NOT NULL DEFAULT 0,
                    user_feedback INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'new',
                    created_at TEXT NOT NULL,
                    delivered_at TEXT,
                    content_type TEXT NOT NULL DEFAULT 'news',
                    arxiv_id TEXT,
                    authors_json TEXT,
                    venue TEXT,
                    citation_count INTEGER,
                    hf_upvotes INTEGER,
                    code_url TEXT,
                    paper_review_json TEXT,
                    bibtex TEXT
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_weights (
                    topic TEXT PRIMARY KEY,
                    category TEXT NOT NULL DEFAULT 'core',
                    weight REAL NOT NULL DEFAULT 1.0,
                    votes INTEGER NOT NULL DEFAULT 0,
                    ignores INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS polls (
                    poll_id TEXT PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER,
                    poll_type TEXT NOT NULL,
                    options_metadata_json TEXT NOT NULL,
                    selected_option_ids_json TEXT,
                    created_at TEXT NOT NULL,
                    answered_at TEXT,
                    user_id INTEGER
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    display_name TEXT,
                    role TEXT NOT NULL DEFAULT 'member',
                    status TEXT NOT NULL DEFAULT 'active',
                    preference_summary TEXT,
                    digest_interval_hours REAL,
                    max_articles_per_digest INTEGER,
                    quiet_hours TEXT,
                    digest_mode TEXT NOT NULL DEFAULT 'balanced',
                    digests_paused INTEGER NOT NULL DEFAULT 0,
                    last_digest_at TEXT,
                    digests_delivered_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_active_at TEXT
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_topic_weights (
                    user_id INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'core',
                    weight REAL NOT NULL DEFAULT 1.0,
                    votes INTEGER NOT NULL DEFAULT 0,
                    ignores INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, topic)
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_deliveries (
                    user_id INTEGER NOT NULL,
                    article_id TEXT NOT NULL,
                    delivered_at TEXT NOT NULL,
                    feedback INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, article_id)
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_library (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    article_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'to_read',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    notes TEXT,
                    saved_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (user_id, article_id)
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS invite_tokens (
                    token_hash TEXT PRIMARY KEY,
                    created_by INTEGER NOT NULL,
                    role TEXT NOT NULL DEFAULT 'member',
                    max_uses INTEGER NOT NULL DEFAULT 1,
                    uses_count INTEGER NOT NULL DEFAULT 0,
                    expires_at TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor_user_id INTEGER,
                    event_type TEXT NOT NULL,
                    target TEXT,
                    details_json TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

            # Safe additive migrations for existing DBs
            self._ensure_column(conn, "feeds", "etag", "TEXT")
            self._ensure_column(conn, "feeds", "last_modified", "TEXT")
            self._ensure_column(conn, "feeds", "last_error", "TEXT")
            self._ensure_column(conn, "feeds", "consecutive_failures", "INTEGER NOT NULL DEFAULT 0")

            self._ensure_column(conn, "articles", "stage1_attempts", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "articles", "user_feedback", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "articles", "content_type", "TEXT NOT NULL DEFAULT 'news'")
            self._ensure_column(conn, "articles", "arxiv_id", "TEXT")
            self._ensure_column(conn, "articles", "authors_json", "TEXT")
            self._ensure_column(conn, "articles", "venue", "TEXT")
            self._ensure_column(conn, "articles", "citation_count", "INTEGER")
            self._ensure_column(conn, "articles", "hf_upvotes", "INTEGER")
            self._ensure_column(conn, "articles", "code_url", "TEXT")
            self._ensure_column(conn, "articles", "paper_review_json", "TEXT")
            self._ensure_column(conn, "articles", "bibtex", "TEXT")

            self._ensure_column(conn, "polls", "message_id", "INTEGER")
            self._ensure_column(conn, "polls", "selected_option_ids_json", "TEXT")
            self._ensure_column(conn, "polls", "user_id", "INTEGER")

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_articles_status_score
                    ON articles (status, relevance_score DESC, published_at DESC, created_at DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_articles_delivered
                    ON articles (status, delivered_at DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_articles_feedback
                    ON articles (user_feedback, delivered_at DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_articles_arxiv_id
                    ON articles (arxiv_id);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_articles_content_type
                    ON articles (content_type, status, relevance_score DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_deliveries_user_ts
                    ON user_deliveries (user_id, delivered_at DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_deliveries_article_fb
                    ON user_deliveries (article_id, feedback);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_library_user_status
                    ON user_library (user_id, status, saved_at DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_log_created
                    ON audit_log (created_at DESC);
                """
            )

            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts
                    USING fts5(article_id UNINDEXED, title, snippet, topics, authors);
                    """
                )
                fts_count = int(conn.execute("SELECT COUNT(*) FROM articles_fts;").fetchone()[0])
                art_count = int(conn.execute("SELECT COUNT(*) FROM articles;").fetchone()[0])
                if fts_count == 0 and art_count > 0:
                    rows = conn.execute(
                        "SELECT id, title, rss_snippet, topics_json, authors_json FROM articles;"
                    ).fetchall()
                    for row in rows:
                        self._upsert_article_fts(
                            conn,
                            row["id"],
                            row["title"] or "",
                            row["rss_snippet"] or "",
                            _format_topics_text(row["topics_json"]),
                            _format_authors_text(row["authors_json"]),
                        )
                self._fts_available = True
            except sqlite3.OperationalError:
                self._fts_available = False

            self._migrate_legacy_single_user_conn(conn)

        harden_file_permissions(self.db_path)

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl};")

    def _upsert_article_fts(
        self,
        conn: sqlite3.Connection,
        article_id: str,
        title: str,
        snippet: str,
        topics_text: str,
        authors_text: str,
    ) -> None:
        if not self._fts_available:
            return
        conn.execute("DELETE FROM articles_fts WHERE article_id = ?;", (article_id,))
        conn.execute(
            "INSERT INTO articles_fts (article_id, title, snippet, topics, authors) VALUES (?, ?, ?, ?, ?);",
            (article_id, title, snippet, topics_text, authors_text),
        )

    def _migrate_legacy_single_user_conn(self, conn: sqlite3.Connection) -> None:
        """Promote legacy single-user state['active_chat_id'] & topic_weights into users tables."""
        row = conn.execute("SELECT value FROM state WHERE key = 'active_chat_id';").fetchone()
        if row is None:
            return
        try:
            legacy_uid = int(row["value"])
        except (TypeError, ValueError):
            return

        existing_user = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?;", (legacy_uid,)
        ).fetchone()
        if existing_user is not None:
            return

        now = utc_now_iso()
        pref_row = conn.execute(
            "SELECT value FROM state WHERE key = 'preference_summary';"
        ).fetchone()
        paused_row = conn.execute(
            "SELECT value FROM state WHERE key = 'digests_paused';"
        ).fetchone()
        count_row = conn.execute(
            "SELECT value FROM state WHERE key = 'digests_delivered_count';"
        ).fetchone()

        conn.execute(
            """
            INSERT INTO users (
                user_id, role, status, preference_summary, digests_paused,
                digests_delivered_count, created_at, last_active_at
            )
            VALUES (?, 'admin', 'active', ?, ?, ?, ?, ?);
            """,
            (
                legacy_uid,
                pref_row["value"] if pref_row else None,
                1 if (paused_row and paused_row["value"] == "1") else 0,
                int(count_row["value"]) if (count_row and str(count_row["value"]).isdigit()) else 0,
                now,
                now,
            ),
        )

        tw_rows = conn.execute("SELECT * FROM topic_weights;").fetchall()
        for tw in tw_rows:
            conn.execute(
                """
                INSERT OR IGNORE INTO user_topic_weights (
                    user_id, topic, category, weight, votes, ignores, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    legacy_uid,
                    tw["topic"],
                    tw["category"],
                    float(tw["weight"]),
                    int(tw["votes"]),
                    int(tw["ignores"]),
                    tw["updated_at"],
                ),
            )

    # ------------------------------------------------------------------
    # Feeds
    # ------------------------------------------------------------------

    def seed_feeds(self, urls: list[str]) -> None:
        """Seed initial RSS feeds without overwriting existing enabled/disabled flags."""
        now = utc_now_iso()
        with self._connect() as conn:
            for raw in urls:
                url = clean_feed_url(raw)
                if not url:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO feeds (url, title, enabled, added_at)
                    VALUES (?, NULL, 1, ?);
                    """,
                    (url, now),
                )

    def seed_core_topics(self, topics: list[str]) -> None:
        """Insert missing legacy core topics with default weight 1.0."""
        now = utc_now_iso()
        with self._connect() as conn:
            for topic in topics:
                cleaned = " ".join(topic.strip().split())
                if not cleaned:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO topic_weights (topic, category, weight, votes, ignores, updated_at)
                    VALUES (?, 'core', 1.0, 0, 0, ?);
                    """,
                    (cleaned, now),
                )

    def get_enabled_feeds(self) -> list[dict[str, Any]]:
        """Return all enabled RSS feeds ordered by addition time."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feeds WHERE enabled = 1 ORDER BY added_at ASC, url ASC;"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_feeds(self) -> list[dict[str, Any]]:
        """Return all RSS feeds including soft-disabled ones."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM feeds ORDER BY added_at ASC, url ASC;").fetchall()
        return [dict(row) for row in rows]

    def add_feed(self, url: str, title: str | None = None) -> bool:
        """Add a feed or re-enable a soft-disabled one. Returns True if newly enabled/added."""
        cleaned = clean_feed_url(url)
        if not cleaned:
            return False
        now = utc_now_iso()
        with self._connect() as conn:
            existing = conn.execute("SELECT enabled FROM feeds WHERE url = ?;", (cleaned,)).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO feeds (url, title, enabled, added_at)
                    VALUES (?, ?, 1, ?);
                    """,
                    (cleaned, title, now),
                )
                return True
            if existing["enabled"] == 0:
                conn.execute(
                    """
                    UPDATE feeds
                    SET enabled = 1,
                        title = COALESCE(?, title),
                        last_error = NULL,
                        consecutive_failures = 0
                    WHERE url = ?;
                    """,
                    (title, cleaned),
                )
                return True
            if title:
                conn.execute(
                    "UPDATE feeds SET title = COALESCE(?, title) WHERE url = ?;",
                    (title, cleaned),
                )
            return False

    def remove_feed(self, url: str) -> bool:
        """Soft-disable an enabled feed so restarts preserve removal across .env seeding."""
        cleaned = clean_feed_url(url)
        if not cleaned:
            return False
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE feeds SET enabled = 0 WHERE url = ? AND enabled = 1;",
                (cleaned,),
            )
            if cur.rowcount == 0 and cleaned.endswith("/"):
                cur = conn.execute(
                    "UPDATE feeds SET enabled = 0 WHERE url = ? AND enabled = 1;",
                    (cleaned.rstrip("/"),),
                )
            elif cur.rowcount == 0 and not cleaned.endswith("/"):
                cur = conn.execute(
                    "UPDATE feeds SET enabled = 0 WHERE url = ? AND enabled = 1;",
                    (cleaned + "/",),
                )
            return cur.rowcount > 0

    def update_feed_checked(
        self,
        url: str,
        etag: str | None = None,
        last_modified: str | None = None,
        title: str | None = None,
    ) -> None:
        """Record successful feed check, reset error counters, and persist HTTP caching headers."""
        cleaned = clean_feed_url(url)
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE feeds
                SET last_checked_at = ?,
                    etag = COALESCE(?, etag),
                    last_modified = COALESCE(?, last_modified),
                    title = COALESCE(?, title),
                    last_error = NULL,
                    consecutive_failures = 0
                WHERE url = ?;
                """,
                (now, etag, last_modified, title, cleaned),
            )

    def record_feed_error(
        self, url: str, error: str, *, max_consecutive_failures: int = 0
    ) -> tuple[int, bool]:
        """Record a feed fetch failure and auto-disable if threshold is reached."""
        cleaned = clean_feed_url(url)
        now = utc_now_iso()
        truncated = (error or "Unknown error").strip()[:500]
        with self._connect() as conn:
            row = conn.execute(
                "SELECT enabled, consecutive_failures FROM feeds WHERE url = ?;",
                (cleaned,),
            ).fetchone()
            if row is None:
                return (0, False)

            new_failures = int(row["consecutive_failures"] or 0) + 1
            auto_disabled = (
                row["enabled"] == 1
                and max_consecutive_failures > 0
                and new_failures >= max_consecutive_failures
            )
            if auto_disabled:
                conn.execute(
                    """
                    UPDATE feeds
                    SET last_checked_at = ?,
                        last_error = ?,
                        consecutive_failures = ?,
                        enabled = 0
                    WHERE url = ?;
                    """,
                    (now, truncated, new_failures, cleaned),
                )
            else:
                conn.execute(
                    """
                    UPDATE feeds
                    SET last_checked_at = ?,
                        last_error = ?,
                        consecutive_failures = ?
                    WHERE url = ?;
                    """,
                    (now, truncated, new_failures, cleaned),
                )
            return (new_failures, auto_disabled)

    # ------------------------------------------------------------------
    # Articles, Papers & FTS5 Search
    # ------------------------------------------------------------------

    def insert_new_articles(self, entries: list[dict[str, Any]]) -> int:
        """Insert unseen articles/papers, backfill comments/paper metadata, and update FTS5."""
        now = utc_now_iso()
        inserted = 0
        with self._connect() as conn:
            for entry in entries:
                article_id = entry["id"]
                comments_url = entry.get("comments_url")
                content_type = entry.get("content_type") or "news"
                arxiv_id = entry.get("arxiv_id")
                authors_json = entry.get("authors_json")
                if authors_json is None and isinstance(entry.get("authors"), list):
                    authors_json = json.dumps(entry["authors"])
                venue = entry.get("venue")
                citation_count = entry.get("citation_count")
                hf_upvotes = entry.get("hf_upvotes")
                code_url = entry.get("code_url")
                bibtex = entry.get("bibtex")
                snippet = entry.get("rss_snippet") or ""

                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO articles (
                        id, url, comments_url, title, source_feed,
                        published_at, rss_snippet, status, created_at,
                        content_type, arxiv_id, authors_json, venue,
                        citation_count, hf_upvotes, code_url, bibtex
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'new', ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        article_id,
                        entry["url"],
                        comments_url,
                        entry["title"],
                        entry["source_feed"],
                        entry.get("published_at") or now,
                        snippet,
                        now,
                        content_type,
                        arxiv_id,
                        authors_json,
                        venue,
                        citation_count,
                        hf_upvotes,
                        code_url,
                        bibtex,
                    ),
                )
                if cur.rowcount > 0:
                    inserted += 1
                    self._upsert_article_fts(
                        conn,
                        article_id,
                        entry["title"],
                        snippet,
                        _format_topics_text(entry.get("topics_json")),
                        _format_authors_text(authors_json),
                    )
                else:
                    if comments_url:
                        conn.execute(
                            """
                            UPDATE articles
                            SET comments_url = ?
                            WHERE id = ? AND comments_url IS NULL;
                            """,
                            (comments_url, article_id),
                        )
                    if hf_upvotes is not None or code_url or citation_count is not None:
                        conn.execute(
                            """
                            UPDATE articles
                            SET hf_upvotes = COALESCE(?, hf_upvotes),
                                code_url = COALESCE(code_url, ?),
                                citation_count = COALESCE(?, citation_count),
                                authors_json = COALESCE(authors_json, ?),
                                arxiv_id = COALESCE(arxiv_id, ?),
                                bibtex = COALESCE(bibtex, ?)
                            WHERE id = ?;
                            """,
                            (
                                hf_upvotes,
                                code_url,
                                citation_count,
                                authors_json,
                                arxiv_id,
                                bibtex,
                                article_id,
                            ),
                        )
        return inserted

    def upsert_paper_article(self, paper: dict[str, Any]) -> str:
        """Insert or enrich a paper record in articles and return its deterministic article_id."""
        now = utc_now_iso()
        article_id = paper["id"]
        authors_json = paper.get("authors_json")
        if authors_json is None and isinstance(paper.get("authors"), list):
            authors_json = json.dumps(paper["authors"])
        topics_json = paper.get("topics_json")
        if topics_json is None and isinstance(paper.get("topics"), list):
            topics_json = json.dumps(paper["topics"])
        snippet = paper.get("rss_snippet") or paper.get("abstract") or ""

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM articles WHERE id = ?;", (article_id,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO articles (
                        id, url, comments_url, title, source_feed, published_at,
                        rss_snippet, full_text, llm_score, relevance_score,
                        relevance_reason, topics_json, status, created_at,
                        content_type, arxiv_id, authors_json, venue,
                        citation_count, hf_upvotes, code_url, bibtex
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'paper', ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        article_id,
                        paper["url"],
                        paper.get("comments_url"),
                        paper["title"],
                        paper.get("source_feed") or "https://arxiv.org",
                        paper.get("published_at") or now,
                        snippet[:800],
                        paper.get("full_text") or snippet,
                        paper.get("llm_score"),
                        paper.get("relevance_score"),
                        paper.get("relevance_reason"),
                        topics_json,
                        paper.get("status") or "queued",
                        now,
                        paper.get("arxiv_id"),
                        authors_json,
                        paper.get("venue"),
                        paper.get("citation_count"),
                        paper.get("hf_upvotes"),
                        paper.get("code_url"),
                        paper.get("bibtex"),
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE articles
                    SET content_type = 'paper',
                        arxiv_id = COALESCE(?, arxiv_id),
                        authors_json = COALESCE(?, authors_json),
                        venue = COALESCE(?, venue),
                        citation_count = COALESCE(?, citation_count),
                        hf_upvotes = COALESCE(?, hf_upvotes),
                        code_url = COALESCE(?, code_url),
                        bibtex = COALESCE(?, bibtex),
                        full_text = COALESCE(full_text, ?)
                    WHERE id = ?;
                    """,
                    (
                        paper.get("arxiv_id"),
                        authors_json,
                        paper.get("venue"),
                        paper.get("citation_count"),
                        paper.get("hf_upvotes"),
                        paper.get("code_url"),
                        paper.get("bibtex"),
                        paper.get("full_text") or snippet or None,
                        article_id,
                    ),
                )

            row = conn.execute(
                "SELECT title, rss_snippet, topics_json, authors_json FROM articles WHERE id = ?;",
                (article_id,),
            ).fetchone()
            if row is not None:
                self._upsert_article_fts(
                    conn,
                    article_id,
                    row["title"] or "",
                    row["rss_snippet"] or "",
                    _format_topics_text(row["topics_json"]),
                    _format_authors_text(row["authors_json"]),
                )
        return article_id

    def get_article_by_arxiv_id(self, arxiv_id: str) -> dict[str, Any] | None:
        """Look up an article by exact or versionless arXiv ID."""
        cleaned = (arxiv_id or "").strip()
        if not cleaned:
            return None
        base_id = re.sub(r"v\d+$", "", cleaned, flags=re.IGNORECASE)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM articles
                WHERE arxiv_id = ? OR arxiv_id LIKE ?
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (cleaned, f"{base_id}%"),
            ).fetchone()
        return dict(row) if row is not None else None

    def save_paper_review(self, article_id: str, review_json: str) -> None:
        """Persist a cached structured PaperReview JSON on the article row."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE articles SET paper_review_json = ? WHERE id = ?;",
                (review_json, article_id),
            )

    def get_cached_paper_review(self, article_id: str) -> str | None:
        """Return cached PaperReview JSON for an article, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT paper_review_json FROM articles WHERE id = ?;",
                (article_id,),
            ).fetchone()
        if row is None:
            return None
        return row["paper_review_json"]

    def search_articles_fts(
        self,
        query: str,
        *,
        content_type: str | None = None,
        limit: int = 10,
        allow_or_fallback: bool = False,
    ) -> list[dict[str, Any]]:
        """Safely search articles via FTS5 (with optional OR fallback & tokenized LIKE fallback)."""
        tokens = _FTS_TOKEN_RE.findall((query or "").strip())
        if not tokens:
            return []

        type_clause = ""
        if content_type in {"news", "paper"}:
            type_clause = " AND a.content_type = ?"

        with self._connect() as conn:
            if self._fts_available:
                for joiner in ((" ", " OR ") if (allow_or_fallback and len(tokens) > 1) else (" ",)):
                    fts_query = joiner.join(f'"{tok}"' for tok in tokens[:8])
                    sql = f"""
                        SELECT a.*
                        FROM articles_fts f
                        JOIN articles a ON a.id = f.article_id
                        WHERE articles_fts MATCH ?{type_clause}
                        ORDER BY bm25(articles_fts), COALESCE(a.relevance_score, 0.5) DESC, a.published_at DESC
                        LIMIT ?;
                    """
                    params: list[Any] = [fts_query]
                    if type_clause:
                        params.append(content_type)
                    params.append(limit)
                    try:
                        rows = conn.execute(sql, params).fetchall()
                        if rows or not allow_or_fallback:
                            return [dict(row) for row in rows]
                    except sqlite3.OperationalError:
                        break
                if allow_or_fallback:
                    return []

            like_clauses = []
            like_params: list[Any] = []
            for tok in tokens[:5]:
                pat = f"%{tok}%"
                like_clauses.append("(a.title LIKE ? OR a.rss_snippet LIKE ? OR COALESCE(a.topics_json, '') LIKE ?)")
                like_params.extend([pat, pat, pat])
            where_sql = " AND ".join(like_clauses)
            if type_clause:
                where_sql += " AND a.content_type = ?"
                like_params.append(content_type)
            like_params.append(limit)
            rows = conn.execute(
                f"""
                SELECT a.* FROM articles a
                WHERE {where_sql}
                ORDER BY COALESCE(a.relevance_score, 0.5) DESC, a.published_at DESC
                LIMIT ?;
                """,
                like_params,
            ).fetchall()
            return [dict(row) for row in rows]

    def get_new_articles(self, limit: int = 60) -> list[dict[str, Any]]:
        """Return newest unscored articles ordered newest-first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM articles
                WHERE status = 'new'
                ORDER BY published_at DESC, created_at DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def increment_stage1_attempts(self, article_ids: list[str], max_attempts: int = 3) -> None:
        """Increment retry counter on transient Stage 1 errors; mark 'failed' after max_attempts."""
        if not article_ids:
            return
        with self._connect() as conn:
            for article_id in article_ids:
                conn.execute(
                    """
                    UPDATE articles
                    SET stage1_attempts = stage1_attempts + 1,
                        status = CASE
                            WHEN stage1_attempts + 1 >= ? THEN 'failed'
                            ELSE status
                        END
                    WHERE id = ? AND status = 'new';
                    """,
                    (max_attempts, article_id),
                )

    def reset_failed_articles(
        self, max_age_hours: int | None = None, *, now: datetime | None = None
    ) -> int:
        """Reset recent 'failed' articles to 'new' (attempts=0), expiring stale ones if max_age_hours is set."""
        cutoff = (
            (now or datetime.now(timezone.utc)) - timedelta(hours=max_age_hours)
            if max_age_hours is not None
            else None
        )
        reset_ids: list[str] = []
        expire_ids: list[str] = []

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, published_at, created_at FROM articles WHERE status = 'failed';"
            ).fetchall()
            for row in rows:
                if cutoff is None:
                    reset_ids.append(row["id"])
                    continue
                ts = parse_iso_utc(row["published_at"]) or parse_iso_utc(row["created_at"])
                if ts is not None and ts < cutoff:
                    expire_ids.append(row["id"])
                else:
                    reset_ids.append(row["id"])

            if reset_ids:
                placeholders = ",".join("?" for _ in reset_ids)
                conn.execute(
                    f"UPDATE articles SET status = 'new', stage1_attempts = 0 WHERE id IN ({placeholders});",
                    reset_ids,
                )
            if expire_ids:
                placeholders = ",".join("?" for _ in expire_ids)
                conn.execute(
                    f"UPDATE articles SET status = 'expired' WHERE id IN ({placeholders});",
                    expire_ids,
                )

        return len(reset_ids)

    def update_article_stage1_score(
        self,
        article_id: str,
        llm_score: float,
        relevance_score: float,
        reason: str,
        topics: list[str],
        status: str,
    ) -> None:
        """Persist Stage 1 relevance scoring, status transition, and refresh FTS5 topics."""
        topics_json = json.dumps(topics)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE articles
                SET llm_score = ?,
                    relevance_score = ?,
                    relevance_reason = ?,
                    topics_json = ?,
                    status = ?
                WHERE id = ?;
                """,
                (
                    llm_score,
                    relevance_score,
                    reason,
                    topics_json,
                    status,
                    article_id,
                ),
            )
            row = conn.execute(
                "SELECT title, rss_snippet, authors_json FROM articles WHERE id = ?;",
                (article_id,),
            ).fetchone()
            if row is not None:
                self._upsert_article_fts(
                    conn,
                    article_id,
                    row["title"] or "",
                    row["rss_snippet"] or "",
                    ", ".join(topics),
                    _format_authors_text(row["authors_json"]),
                )

    def expire_old_queued_articles(
        self, max_age_hours: int, *, now: datetime | None = None
    ) -> int:
        """Mark queued articles older than max_age_hours as 'expired' and return count."""
        ref_now = now or datetime.now(timezone.utc)
        cutoff = ref_now - timedelta(hours=max_age_hours)
        expired_ids: list[str] = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, published_at, created_at FROM articles WHERE status = 'queued';"
            ).fetchall()
            for row in rows:
                ts = parse_iso_utc(row["published_at"]) or parse_iso_utc(row["created_at"])
                if ts is not None and ts < cutoff:
                    expired_ids.append(row["id"])
            if expired_ids:
                placeholders = ",".join("?" for _ in expired_ids)
                conn.execute(
                    f"UPDATE articles SET status = 'expired' WHERE id IN ({placeholders});",
                    expired_ids,
                )
        return len(expired_ids)

    def prune_old_articles(
        self,
        retention_days: int,
        *,
        min_age_hours: int = 0,
        now: datetime | None = None,
    ) -> int:
        """Delete terminal non-delivered articles older than retention window (protecting library/deliveries)."""
        if retention_days <= 0:
            return 0
        ref_now = now or datetime.now(timezone.utc)
        effective_delta = max(
            timedelta(days=retention_days),
            timedelta(hours=max(0, min_age_hours)),
        )
        cutoff = ref_now - effective_delta
        prune_ids: list[str] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, published_at, created_at
                FROM articles
                WHERE status IN ('expired', 'filtered_out', 'failed')
                  AND id NOT IN (SELECT article_id FROM user_library)
                  AND id NOT IN (SELECT article_id FROM user_deliveries);
                """
            ).fetchall()
            for row in rows:
                timestamps = [
                    ts
                    for ts in (
                        parse_iso_utc(row["published_at"]),
                        parse_iso_utc(row["created_at"]),
                    )
                    if ts is not None
                ]
                if not timestamps:
                    continue
                if max(timestamps) < cutoff:
                    prune_ids.append(row["id"])

            for start in range(0, len(prune_ids), 500):
                batch = prune_ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                conn.execute(
                    f"DELETE FROM articles WHERE id IN ({placeholders});",
                    batch,
                )
                if self._fts_available:
                    conn.execute(
                        f"DELETE FROM articles_fts WHERE article_id IN ({placeholders});",
                        batch,
                    )
        return len(prune_ids)

    def get_queued_articles(self, limit: int) -> list[dict[str, Any]]:
        """Return queued articles ordered by relevance score and recency."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM articles
                WHERE status = 'queued'
                ORDER BY relevance_score DESC, published_at DESC, created_at DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_candidate_articles_for_user(
        self,
        user_id: int,
        *,
        max_age_hours: int,
        content_mode: str = "balanced",
        limit: int = 40,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent relevant articles/papers not yet delivered to user_id."""
        ref_now = now or datetime.now(timezone.utc)
        cutoff = (ref_now - timedelta(hours=max_age_hours)).isoformat()
        mode_clause = ""
        params: list[Any] = [user_id, cutoff]
        if content_mode == "papers":
            mode_clause = "AND a.content_type = 'paper'"
        elif content_mode == "news":
            mode_clause = "AND a.content_type = 'news'"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT a.*
                FROM articles a
                WHERE a.status IN ('queued', 'delivered')
                  AND a.id NOT IN (
                      SELECT article_id FROM user_deliveries WHERE user_id = ?
                  )
                  AND COALESCE(a.published_at, a.created_at) >= ?
                  {mode_clause}
                ORDER BY COALESCE(a.relevance_score, a.llm_score, 0.7) DESC,
                         a.published_at DESC,
                         a.created_at DESC
                LIMIT ?;
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_article(self, article_id: str) -> dict[str, Any] | None:
        """Return a single article row as a dict, or None if not found."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM articles WHERE id = ?;", (article_id,)).fetchone()
        return dict(row) if row is not None else None

    def get_article_by_prefix(self, id_prefix: str) -> dict[str, Any] | None:
        """Return a single article matching a >=6 char hex ID prefix."""
        cleaned = (id_prefix or "").strip().lower()
        if len(cleaned) < 6 or not re.fullmatch(r"[0-9a-f]+", cleaned):
            return None
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM articles WHERE id LIKE ? LIMIT 2;",
                (f"{cleaned}%",),
            ).fetchall()
        if len(rows) == 1:
            return dict(rows[0])
        return None

    def cache_article_summary(
        self,
        article_id: str,
        full_text: str | None,
        hn_comments: str | None,
        summary_json: str,
        topics: list[str],
    ) -> None:
        """Cache Stage 2 enrichment & summary on articles row so all org users share it."""
        now = utc_now_iso()
        topics_json = json.dumps(topics)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE articles
                SET full_text = COALESCE(?, full_text),
                    hn_comments = COALESCE(?, hn_comments),
                    summary_json = ?,
                    topics_json = ?,
                    status = 'delivered',
                    delivered_at = COALESCE(delivered_at, ?)
                WHERE id = ?;
                """,
                (full_text, hn_comments, summary_json, topics_json, now, article_id),
            )
            row = conn.execute(
                "SELECT title, rss_snippet, authors_json FROM articles WHERE id = ?;",
                (article_id,),
            ).fetchone()
            if row is not None:
                self._upsert_article_fts(
                    conn,
                    article_id,
                    row["title"] or "",
                    row["rss_snippet"] or "",
                    ", ".join(topics),
                    _format_authors_text(row["authors_json"]),
                )

    def mark_article_delivered(
        self,
        article_id: str,
        full_text: str | None,
        hn_comments: str | None,
        summary_json: str,
        topics: list[str],
    ) -> None:
        """Mark an article delivered after Telegram message delivery succeeds."""
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE articles
                SET full_text = ?,
                    hn_comments = ?,
                    summary_json = ?,
                    topics_json = ?,
                    status = 'delivered',
                    delivered_at = ?
                WHERE id = ?;
                """,
                (
                    full_text,
                    hn_comments,
                    summary_json,
                    json.dumps(topics),
                    now,
                    article_id,
                ),
            )

    def record_user_deliveries(self, user_id: int, article_ids: list[str]) -> None:
        """Record per-user article deliveries so each org user receives stories once."""
        if not article_ids:
            return
        now = utc_now_iso()
        with self._connect() as conn:
            for art_id in article_ids:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO user_deliveries (user_id, article_id, delivered_at, feedback)
                    VALUES (?, ?, ?, 0);
                    """,
                    (user_id, art_id, now),
                )

    def set_articles_feedback(self, liked_ids: list[str], skipped_ids: list[str]) -> None:
        """Persist explicit poll vote feedback (+1 liked, -1 skipped) on delivered articles."""
        with self._connect() as conn:
            for art_id in liked_ids:
                conn.execute(
                    "UPDATE articles SET user_feedback = 1 WHERE id = ?;",
                    (art_id,),
                )
            for art_id in skipped_ids:
                conn.execute(
                    "UPDATE articles SET user_feedback = -1 WHERE id = ?;",
                    (art_id,),
                )

    def set_user_articles_feedback(
        self, user_id: int, liked_ids: list[str], skipped_ids: list[str]
    ) -> None:
        """Persist per-user poll vote feedback in user_deliveries."""
        now = utc_now_iso()
        with self._connect() as conn:
            for art_id in liked_ids:
                conn.execute(
                    """
                    INSERT INTO user_deliveries (user_id, article_id, delivered_at, feedback)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(user_id, article_id) DO UPDATE SET feedback = 1;
                    """,
                    (user_id, art_id, now),
                )
            for art_id in skipped_ids:
                conn.execute(
                    """
                    INSERT INTO user_deliveries (user_id, article_id, delivered_at, feedback)
                    VALUES (?, ?, ?, -1)
                    ON CONFLICT(user_id, article_id) DO UPDATE SET feedback = -1;
                    """,
                    (user_id, art_id, now),
                )

    def get_recent_titles_for_profile(self, limit: int = 10) -> tuple[list[str], list[str]]:
        """Return recent liked and skipped article titles based on articles.user_feedback."""
        with self._connect() as conn:
            liked_rows = conn.execute(
                """
                SELECT title FROM articles
                WHERE user_feedback = 1
                ORDER BY delivered_at DESC, created_at DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()
            skipped_rows = conn.execute(
                """
                SELECT title FROM articles
                WHERE user_feedback = -1
                ORDER BY delivered_at DESC, created_at DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()
        return (
            [row["title"] for row in liked_rows],
            [row["title"] for row in skipped_rows],
        )

    def get_user_recent_titles_for_profile(
        self, user_id: int, limit: int = 10
    ) -> tuple[list[str], list[str]]:
        """Return recent liked and skipped article titles for a specific org user."""
        with self._connect() as conn:
            liked_rows = conn.execute(
                """
                SELECT a.title FROM user_deliveries d
                JOIN articles a ON a.id = d.article_id
                WHERE d.user_id = ? AND d.feedback = 1
                ORDER BY d.delivered_at DESC
                LIMIT ?;
                """,
                (user_id, limit),
            ).fetchall()
            skipped_rows = conn.execute(
                """
                SELECT a.title FROM user_deliveries d
                JOIN articles a ON a.id = d.article_id
                WHERE d.user_id = ? AND d.feedback = -1
                ORDER BY d.delivered_at DESC
                LIMIT ?;
                """,
                (user_id, limit),
            ).fetchall()
        return (
            [row["title"] for row in liked_rows],
            [row["title"] for row in skipped_rows],
        )

    def get_recent_ai_titles(self, limit: int = 30) -> list[str]:
        """Return recent AI article titles (queued or delivered) for discovery proposals."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT title FROM articles
                WHERE status IN ('queued', 'delivered')
                ORDER BY COALESCE(delivered_at, created_at) DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()
        return [row["title"] for row in rows]

    def get_recent_papers(self, days: int = 7, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent academic papers ordered by HF upvotes, citations, and recency."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM articles
                WHERE content_type = 'paper'
                  AND COALESCE(published_at, created_at) >= ?
                ORDER BY COALESCE(hf_upvotes, 0) DESC,
                         COALESCE(citation_count, 0) DESC,
                         COALESCE(relevance_score, 0.0) DESC,
                         published_at DESC
                LIMIT ?;
                """,
                (cutoff, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_org_trending_articles(self, days: int = 7, limit: int = 10) -> list[dict[str, Any]]:
        """Return anonymized org-wide most-upvoted & most-saved articles/papers."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT a.*,
                       COALESCE(SUM(CASE WHEN d.feedback = 1 THEN 1 ELSE 0 END), 0) AS org_likes,
                       (SELECT COUNT(*) FROM user_library l WHERE l.article_id = a.id) AS org_saves
                FROM articles a
                LEFT JOIN user_deliveries d ON d.article_id = a.id
                WHERE COALESCE(a.published_at, a.created_at) >= ?
                  AND a.status IN ('queued', 'delivered')
                GROUP BY a.id
                HAVING (org_likes + org_saves) > 0
                ORDER BY (org_likes * 2 + org_saves * 3) DESC,
                         COALESCE(a.hf_upvotes, 0) DESC,
                         a.published_at DESC
                LIMIT ?;
                """,
                (cutoff, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Legacy Global + Per-User Topic Weights
    # ------------------------------------------------------------------

    def get_all_topic_weights(self) -> list[dict[str, Any]]:
        """Return all legacy topic weights ordered highest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM topic_weights
                ORDER BY weight DESC, votes DESC, topic ASC;
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_topic_weight(
        self,
        topic: str,
        category: str,
        new_weight: float,
        vote_delta: int = 0,
        ignore_delta: int = 0,
    ) -> None:
        """Insert or update a legacy topic weight while preserving core category status."""
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO topic_weights (topic, category, weight, votes, ignores, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic) DO UPDATE SET
                    weight = excluded.weight,
                    votes = topic_weights.votes + excluded.votes,
                    ignores = topic_weights.ignores + excluded.ignores,
                    updated_at = excluded.updated_at;
                """,
                (topic, category, new_weight, vote_delta, ignore_delta, now),
            )

    def apply_weight_mean_reversion(self, factor: float = 0.98) -> None:
        """Gently nudge all legacy topic weights toward neutral 1.0 to prevent saturation."""
        now = utc_now_iso()
        with self._connect() as conn:
            rows = conn.execute("SELECT topic, weight FROM topic_weights;").fetchall()
            for row in rows:
                w = float(row["weight"])
                new_w = round(max(0.1, min(3.0, 1.0 + factor * (w - 1.0))), 4)
                if new_w != w:
                    conn.execute(
                        "UPDATE topic_weights SET weight = ?, updated_at = ? WHERE topic = ?;",
                        (new_w, now, row["topic"]),
                    )

    def seed_user_core_topics(self, user_id: int, topics: list[str]) -> None:
        """Insert missing core topics for a specific user with default weight 1.0."""
        now = utc_now_iso()
        with self._connect() as conn:
            for topic in topics:
                cleaned = " ".join(topic.strip().split())
                if not cleaned:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO user_topic_weights (
                        user_id, topic, category, weight, votes, ignores, updated_at
                    )
                    VALUES (?, ?, 'core', 1.0, 0, 0, ?);
                    """,
                    (user_id, cleaned, now),
                )

    def get_user_topic_weights(self, user_id: int) -> list[dict[str, Any]]:
        """Return topic weights for a specific user ordered highest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM user_topic_weights
                WHERE user_id = ?
                ORDER BY weight DESC, votes DESC, topic ASC;
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_user_topic_weight(
        self,
        user_id: int,
        topic: str,
        category: str,
        new_weight: float,
        vote_delta: int = 0,
        ignore_delta: int = 0,
    ) -> None:
        """Insert or update a per-user topic weight."""
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_topic_weights (
                    user_id, topic, category, weight, votes, ignores, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, topic) DO UPDATE SET
                    weight = excluded.weight,
                    votes = user_topic_weights.votes + excluded.votes,
                    ignores = user_topic_weights.ignores + excluded.ignores,
                    updated_at = excluded.updated_at;
                """,
                (user_id, topic, category, new_weight, vote_delta, ignore_delta, now),
            )

    def apply_user_weight_mean_reversion(self, user_id: int, factor: float = 0.98) -> None:
        """Gently nudge a user's topic weights toward neutral 1.0."""
        now = utc_now_iso()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT topic, weight FROM user_topic_weights WHERE user_id = ?;",
                (user_id,),
            ).fetchall()
            for row in rows:
                w = float(row["weight"])
                new_w = round(max(0.1, min(3.0, 1.0 + factor * (w - 1.0))), 4)
                if new_w != w:
                    conn.execute(
                        """
                        UPDATE user_topic_weights
                        SET weight = ?, updated_at = ?
                        WHERE user_id = ? AND topic = ?;
                        """,
                        (new_w, now, user_id, row["topic"]),
                    )

    # ------------------------------------------------------------------
    # Users, RBAC, Invites & Preferences
    # ------------------------------------------------------------------

    def upsert_user(
        self,
        user_id: int,
        *,
        username: str | None = None,
        display_name: str | None = None,
        role: str = "member",
        status: str = "active",
        core_topics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create or update a user record and optionally seed core topics."""
        if role not in _VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")
        if status not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        now = utc_now_iso()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM users WHERE user_id = ?;", (user_id,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO users (
                        user_id, username, display_name, role, status,
                        created_at, last_active_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (user_id, username, display_name, role, status, now, now),
                )
            else:
                conn.execute(
                    """
                    UPDATE users
                    SET username = COALESCE(?, username),
                        display_name = COALESCE(?, display_name),
                        role = ?,
                        status = ?,
                        last_active_at = ?
                    WHERE user_id = ?;
                    """,
                    (username, display_name, role, status, now, user_id),
                )
        if core_topics:
            self.seed_user_core_topics(user_id, core_topics)
        user = self.get_user(user_id)
        assert user is not None
        return user

    def seed_users_from_settings(
        self,
        admin_ids: list[int],
        member_ids: list[int],
        core_topics: list[str],
    ) -> None:
        """Ensure env-configured admins and allowed members exist and are active."""
        for uid in admin_ids:
            existing = self.get_user(uid)
            if existing is None:
                self.upsert_user(uid, role="admin", status="active", core_topics=core_topics)
            else:
                if existing["role"] != "admin" or existing["status"] != "active":
                    self.upsert_user(uid, role="admin", status="active", core_topics=core_topics)
                else:
                    self.seed_user_core_topics(uid, core_topics)

        admin_set = set(admin_ids)
        for uid in member_ids:
            if uid in admin_set:
                continue
            existing = self.get_user(uid)
            if existing is None:
                self.upsert_user(uid, role="member", status="active", core_topics=core_topics)
            else:
                if existing["status"] != "active":
                    self.set_user_status(uid, "active")
                self.seed_user_core_topics(uid, core_topics)

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        """Fetch a user row by Telegram user_id."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?;", (user_id,)).fetchone()
        return dict(row) if row is not None else None

    def list_users(self, status: str | None = None) -> list[dict[str, Any]]:
        """List org users, optionally filtered by status."""
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM users WHERE status = ? ORDER BY role ASC, created_at ASC;",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM users ORDER BY status ASC, role ASC, created_at ASC;"
                ).fetchall()
        return [dict(row) for row in rows]

    def count_active_users(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM users WHERE status = 'active';").fetchone()
        return int(row[0])

    def count_admins(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'admin' AND status = 'active';"
            ).fetchone()
        return int(row[0])

    def set_user_role(self, user_id: int, role: str) -> bool:
        if role not in _VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")
        with self._connect() as conn:
            cur = conn.execute("UPDATE users SET role = ? WHERE user_id = ?;", (role, user_id))
        return cur.rowcount > 0

    def set_user_status(self, user_id: int, status: str) -> bool:
        if status not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        with self._connect() as conn:
            cur = conn.execute("UPDATE users SET status = ? WHERE user_id = ?;", (status, user_id))
        return cur.rowcount > 0

    def touch_user_active(
        self, user_id: int, username: str | None = None, display_name: str | None = None
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET last_active_at = ?,
                    username = COALESCE(?, username),
                    display_name = COALESCE(?, display_name)
                WHERE user_id = ?;
                """,
                (now, username, display_name, user_id),
            )

    def get_user_preference_summary(self, user_id: int) -> str | None:
        user = self.get_user(user_id)
        if user and user.get("preference_summary"):
            return str(user["preference_summary"])
        return None

    def set_user_preference_summary(self, user_id: int, summary: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET preference_summary = ? WHERE user_id = ?;",
                (summary, user_id),
            )

    def update_user_settings(
        self,
        user_id: int,
        *,
        digest_interval_hours: float | None = None,
        max_articles_per_digest: int | None = None,
        quiet_hours: str | None = ...,  # type: ignore[assignment]
        digest_mode: str | None = None,
        digests_paused: int | None = None,
    ) -> bool:
        updates: list[str] = []
        params: list[Any] = []
        if digest_interval_hours is not None:
            updates.append("digest_interval_hours = ?")
            params.append(float(digest_interval_hours))
        if max_articles_per_digest is not None:
            updates.append("max_articles_per_digest = ?")
            params.append(max(1, min(9, int(max_articles_per_digest))))
        if quiet_hours is not ...:
            updates.append("quiet_hours = ?")
            params.append(quiet_hours)
        if digest_mode is not None:
            updates.append("digest_mode = ?")
            params.append(digest_mode)
        if digests_paused is not None:
            updates.append("digests_paused = ?")
            params.append(int(digests_paused))
        if not updates:
            return False
        params.append(user_id)
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?;",
                params,
            )
        return cur.rowcount > 0

    def increment_user_digest_count(self, user_id: int) -> int:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET digests_delivered_count = COALESCE(digests_delivered_count, 0) + 1,
                    last_digest_at = ?
                WHERE user_id = ?;
                """,
                (now, user_id),
            )
            row = conn.execute(
                "SELECT digests_delivered_count FROM users WHERE user_id = ?;",
                (user_id,),
            ).fetchone()
        return int(row["digests_delivered_count"]) if row else 0

    def get_due_digest_users(
        self, default_interval_hours: float, *, now: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Return active unpaused users whose personal digest interval has elapsed."""
        ref_now = now or datetime.now(timezone.utc)
        due: list[dict[str, Any]] = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM users WHERE status = 'active' AND digests_paused = 0;"
            ).fetchall()
        for row in rows:
            user = dict(row)
            interval_h = float(user.get("digest_interval_hours") or default_interval_hours)
            last_dt = parse_iso_utc(user.get("last_digest_at"))
            if last_dt is None or (ref_now - last_dt).total_seconds() >= interval_h * 3600.0:
                due.append(user)
        return due

    def create_invite_token(
        self,
        token_hash: str,
        created_by: int,
        *,
        role: str = "member",
        max_uses: int = 1,
        expires_hours: int = 72,
    ) -> None:
        if role not in _VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")
        now_dt = datetime.now(timezone.utc)
        expires_at = (now_dt + timedelta(hours=max(1, expires_hours))).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO invite_tokens (
                    token_hash, created_by, role, max_uses, uses_count, expires_at, revoked, created_at
                )
                VALUES (?, ?, ?, ?, 0, ?, 0, ?);
                """,
                (token_hash, created_by, role, max(1, max_uses), expires_at, now_dt.isoformat()),
            )

    def redeem_invite_token(
        self,
        token_hash: str,
        user_id: int,
        *,
        username: str | None = None,
        display_name: str | None = None,
        core_topics: list[str] | None = None,
        now: datetime | None = None,
    ) -> tuple[bool, str]:
        """Atomically validate and redeem an invite token, activating user_id."""
        ref_now = now or datetime.now(timezone.utc)
        now_iso = ref_now.isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM invite_tokens WHERE token_hash = ?;", (token_hash,)
            ).fetchone()
            if row is None:
                return (False, "Invalid invite code.")
            if int(row["revoked"] or 0) == 1:
                return (False, "This invite code has been revoked.")
            expires_dt = parse_iso_utc(row["expires_at"])
            if expires_dt is not None and ref_now > expires_dt:
                return (False, "This invite code has expired.")
            if int(row["uses_count"]) >= int(row["max_uses"]):
                return (False, "This invite code has reached its maximum uses.")

            cur = conn.execute(
                """
                UPDATE invite_tokens
                SET uses_count = uses_count + 1
                WHERE token_hash = ? AND revoked = 0 AND uses_count < max_uses;
                """,
                (token_hash,),
            )
            if cur.rowcount != 1:
                return (False, "Invite code could not be claimed.")

            assigned_role = row["role"] if row["role"] in _VALID_ROLES else "member"
            existing = conn.execute(
                "SELECT role, status FROM users WHERE user_id = ?;", (user_id,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO users (
                        user_id, username, display_name, role, status, created_at, last_active_at
                    )
                    VALUES (?, ?, ?, ?, 'active', ?, ?);
                    """,
                    (user_id, username, display_name, assigned_role, now_iso, now_iso),
                )
            else:
                final_role = "admin" if existing["role"] == "admin" else assigned_role
                conn.execute(
                    """
                    UPDATE users
                    SET username = COALESCE(?, username),
                        display_name = COALESCE(?, display_name),
                        role = ?,
                        status = 'active',
                        last_active_at = ?
                    WHERE user_id = ?;
                    """,
                    (username, display_name, final_role, now_iso, user_id),
                )

        if core_topics:
            self.seed_user_core_topics(user_id, core_topics)
        return (True, assigned_role)

    def revoke_all_invites(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("UPDATE invite_tokens SET revoked = 1 WHERE revoked = 0;")
        return cur.rowcount

    # ------------------------------------------------------------------
    # Reading Library
    # ------------------------------------------------------------------

    def save_to_library(
        self,
        user_id: int,
        article_id: str,
        *,
        tags: list[str] | None = None,
        status: str = "to_read",
    ) -> dict[str, Any]:
        if status not in _VALID_LIBRARY_STATUSES:
            status = "to_read"
        clean_tags = [" ".join(str(t).split())[:40] for t in (tags or []) if str(t).strip()][:8]
        now = utc_now_iso()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM user_library WHERE user_id = ? AND article_id = ?;",
                (user_id, article_id),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO user_library (
                        user_id, article_id, status, tags_json, saved_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (user_id, article_id, status, json.dumps(clean_tags), now, now),
                )
            elif clean_tags:
                old_tags = json.loads(existing["tags_json"] or "[]")
                merged = list(dict.fromkeys([*old_tags, *clean_tags]))[:8]
                conn.execute(
                    "UPDATE user_library SET tags_json = ?, updated_at = ? WHERE id = ?;",
                    (json.dumps(merged), now, existing["id"]),
                )
            row = conn.execute(
                """
                SELECT l.*, a.title, a.url, a.arxiv_id, a.content_type, a.bibtex
                FROM user_library l
                JOIN articles a ON a.id = l.article_id
                WHERE l.user_id = ? AND l.article_id = ?;
                """,
                (user_id, article_id),
            ).fetchone()
        assert row is not None
        return dict(row)

    def _resolve_library_row_id(
        self, conn: sqlite3.Connection, user_id: int, ref: str | int
    ) -> int | None:
        text = str(ref).strip()
        if text.isdigit():
            row = conn.execute(
                "SELECT id FROM user_library WHERE user_id = ? AND id = ?;",
                (user_id, int(text)),
            ).fetchone()
            if row is not None:
                return int(row["id"])
        row = conn.execute(
            """
            SELECT l.id FROM user_library l
            JOIN articles a ON a.id = l.article_id
            WHERE l.user_id = ? AND (l.article_id = ? OR a.arxiv_id = ?);
            """,
            (user_id, text, text),
        ).fetchone()
        return int(row["id"]) if row is not None else None

    def get_library_item(self, user_id: int, ref: str | int) -> dict[str, Any] | None:
        with self._connect() as conn:
            lib_id = self._resolve_library_row_id(conn, user_id, ref)
            if lib_id is None:
                return None
            row = conn.execute(
                """
                SELECT l.*, a.title, a.url, a.arxiv_id, a.content_type, a.authors_json, a.bibtex
                FROM user_library l
                JOIN articles a ON a.id = l.article_id
                WHERE l.id = ? AND l.user_id = ?;
                """,
                (lib_id, user_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def update_library_status(self, user_id: int, ref: str | int, status: str) -> bool:
        if status not in _VALID_LIBRARY_STATUSES:
            raise ValueError(f"Invalid library status: {status}")
        now = utc_now_iso()
        with self._connect() as conn:
            lib_id = self._resolve_library_row_id(conn, user_id, ref)
            if lib_id is None:
                return False
            cur = conn.execute(
                "UPDATE user_library SET status = ?, updated_at = ? WHERE id = ? AND user_id = ?;",
                (status, now, lib_id, user_id),
            )
        return cur.rowcount > 0

    def update_library_notes(self, user_id: int, ref: str | int, notes: str) -> bool:
        now = utc_now_iso()
        cleaned = (notes or "").strip()[:2000]
        with self._connect() as conn:
            lib_id = self._resolve_library_row_id(conn, user_id, ref)
            if lib_id is None:
                return False
            cur = conn.execute(
                "UPDATE user_library SET notes = ?, updated_at = ? WHERE id = ? AND user_id = ?;",
                (cleaned, now, lib_id, user_id),
            )
        return cur.rowcount > 0

    def remove_from_library(self, user_id: int, ref: str | int) -> bool:
        with self._connect() as conn:
            lib_id = self._resolve_library_row_id(conn, user_id, ref)
            if lib_id is None:
                return False
            cur = conn.execute(
                "DELETE FROM user_library WHERE id = ? AND user_id = ?;",
                (lib_id, user_id),
            )
        return cur.rowcount > 0

    def get_user_library(
        self,
        user_id: int,
        *,
        status: str | None = None,
        tag: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        params: list[Any] = [user_id]
        clauses = ["l.user_id = ?"]
        if status and status in _VALID_LIBRARY_STATUSES:
            clauses.append("l.status = ?")
            params.append(status)
        if tag:
            clauses.append("LOWER(l.tags_json) LIKE ?")
            params.append(f'%"{tag.strip().lower()}"%')
        params.append(limit)
        where_sql = " AND ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT l.*, a.title, a.url, a.arxiv_id, a.content_type,
                       a.authors_json, a.published_at, a.venue, a.citation_count,
                       a.hf_upvotes, a.code_url, a.bibtex, a.topics_json
                FROM user_library l
                JOIN articles a ON a.id = l.article_id
                WHERE {where_sql}
                ORDER BY l.updated_at DESC, l.id DESC
                LIMIT ?;
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Polls, Audit Log, Daily Budget & State
    # ------------------------------------------------------------------

    def save_poll(
        self,
        poll_id: str,
        chat_id: int,
        poll_type: str,
        options_metadata: list[dict[str, Any]],
        message_id: int | None = None,
        user_id: int | None = None,
    ) -> None:
        """Persist poll metadata keyed by Telegram poll_id."""
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO polls (
                    poll_id, chat_id, message_id, poll_type,
                    options_metadata_json, selected_option_ids_json, created_at, answered_at, user_id
                )
                VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, ?);
                """,
                (poll_id, chat_id, message_id, poll_type, json.dumps(options_metadata), now, user_id),
            )

    def get_poll(self, poll_id: str) -> dict[str, Any] | None:
        """Return poll record with decoded options_metadata list, or None."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM polls WHERE poll_id = ?;", (poll_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["options_metadata"] = json.loads(record["options_metadata_json"])
        return record

    def claim_poll_answer(self, poll_id: str, option_ids: list[int]) -> bool:
        """Atomically claim a poll for processing once, preventing re-vote double-counting."""
        now = utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE polls
                SET answered_at = ?,
                    selected_option_ids_json = ?
                WHERE poll_id = ? AND answered_at IS NULL;
                """,
                (now, json.dumps(option_ids), poll_id),
            )
            return cur.rowcount == 1

    def mark_poll_answered(self, poll_id: str) -> None:
        """Mark a poll answered if not already timestamped."""
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE polls SET answered_at = COALESCE(answered_at, ?) WHERE poll_id = ?;",
                (now, poll_id),
            )

    def log_audit_event(
        self,
        actor_user_id: int | None,
        event_type: str,
        target: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (actor_user_id, event_type, target, details_json, created_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    actor_user_id,
                    event_type,
                    target[:200] if target else None,
                    json.dumps(details) if details else None,
                    now,
                ),
            )

    def get_recent_audit_events(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?;",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_daily_llm_usage(self, *, now: datetime | None = None) -> int:
        day_key = f"llm_budget_{(now or datetime.now(timezone.utc)).strftime('%Y-%m-%d')}"
        raw = self.get_state(day_key, "0")
        try:
            return int(raw or 0)
        except ValueError:
            return 0

    def consume_daily_llm_budget(
        self, limit: int, count: int = 1, *, now: datetime | None = None
    ) -> tuple[bool, int]:
        """Atomically consume daily org LLM budget. Returns (allowed, new_usage)."""
        ref_now = now or datetime.now(timezone.utc)
        day_key = f"llm_budget_{ref_now.strftime('%Y-%m-%d')}"
        now_iso = ref_now.isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO state (key, value, updated_at) VALUES (?, '0', ?);",
                (day_key, now_iso),
            )
            row = conn.execute(
                "SELECT value FROM state WHERE key = ?;", (day_key,)
            ).fetchone()
            current = int(row["value"] or 0) if row else 0
            if current + count > limit:
                return (False, current)
            new_val = current + count
            conn.execute(
                "UPDATE state SET value = ?, updated_at = ? WHERE key = ?;",
                (str(new_val), now_iso, day_key),
            )
            return (True, new_val)

    def get_state(self, key: str, default: str | None = None) -> str | None:
        """Fetch a key from the state table."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM state WHERE key = ?;", (key,)).fetchone()
        return row["value"] if row is not None else default

    def set_state(self, key: str, value: str) -> None:
        """Upsert a key-value string in the state table."""
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at;
                """,
                (key, value, now),
            )

    def get_stats(self) -> dict[str, Any]:
        """Return operational statistics across articles, papers, feeds, and users."""
        status_counts = {
            "new": 0,
            "queued": 0,
            "filtered_out": 0,
            "expired": 0,
            "failed": 0,
            "delivered": 0,
        }
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM articles GROUP BY status;"
            ).fetchall()
            for row in rows:
                status_counts[row["status"]] = int(row["cnt"])

            paper_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM articles WHERE content_type = 'paper';"
                ).fetchone()[0]
            )
            feed_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_feeds,
                    SUM(CASE WHEN enabled = 1 THEN 1 ELSE 0 END) AS enabled_feeds,
                    SUM(CASE WHEN enabled = 1 AND consecutive_failures > 0 THEN 1 ELSE 0 END) AS failing_feeds
                FROM feeds;
                """
            ).fetchone()
            active_users = int(
                conn.execute("SELECT COUNT(*) FROM users WHERE status = 'active';").fetchone()[0]
            )

        return {
            "status_counts": status_counts,
            "paper_count": paper_count,
            "total_feeds": int(feed_row["total_feeds"] or 0),
            "enabled_feeds": int(feed_row["enabled_feeds"] or 0),
            "failing_feeds": int(feed_row["failing_feeds"] or 0),
            "active_users": active_users,
            "last_scrape_at": self.get_state("last_scrape_at"),
            "digests_delivered_count": int(self.get_state("digests_delivered_count", "0") or 0),
            "digests_paused": self.get_state("digests_paused", "0") == "1",
            "daily_llm_usage": self.get_daily_llm_usage(),
        }
