# DailyAI Telegram Literature Review & Daily Digest Bot

Production-hardened Python 3.11–3.13 Telegram service built for **20+ AI professionals** in an organization. It combines a personalized AI news & Hugging Face / arXiv daily paper digest with structured peer-review analysis, grounded multi-paper literature reviews, BibTeX/Markdown exports, and defense-in-depth cybersecurity controls.

## Key Capabilities

- **Personalized Multi-User Daily Digests (`/digest`, `/topics`, `/poll`, `/discover`, `/settings`)**:
  - Shared Stage 1 ingestion (RSS + Hugging Face Daily Papers) and shared Stage 2 summary caching (`articles.summary_json`) so 20+ users never trigger duplicate scrapes or duplicate LLM calls.
  - Per-user topic weights (`user_topic_weights`), delivery tracking (`user_deliveries`), digest cadence, item count, content mode (`balanced | papers | news`), and personal `quiet_hours`.
- **AI Literature Review Suite (`/review`, `/litreview`, `/compare`, `/search`, `/trending`)**:
  - **`/review <arxiv_id|url|lib_id>`**: Peer-review style breakdown (TL;DR, problem, methodology, quantitative results, explicit limitations, maturity level, reproducibility), cached org-wide.
  - **`/litreview <question>`**: Searches local FTS5 corpus + live arXiv API + Semantic Scholar, synthesizes a grounded report citing `[P1]..[Pn]`, and attaches downloadable `litreview.md` and `references.bib`.
  - **`/compare <id1> <id2> [id3] [id4]`**: Side-by-side comparison across technical dimensions.
  - **`/search [--arxiv] <query>`** & **`/trending`**: Fast FTS5 + arXiv search, Hugging Face Daily Papers, and anonymized org top picks.
- **Personal Research Library & BibTeX (`/library`, `/save`, `/note`, `/read`, `/bibtex`, `/export`)**:
  - Save papers/articles, tag them, add reading notes, track `to_read` / `read`, and export `library.bib` or `library.md`.
- **Cybersecurity & Org Governance (`/admin`, `/join`, `/feeds`)**:
  - **RBAC (`admin` vs `member`)** in 1:1 private DMs, cryptographic SHA-256-hashed invite codes (`/admin invite` -> `/join <code>`), and full `audit_log`.
  - **SSRF & DNS Rebinding Defense**: `SafeNetworkBackend` resolves and validates all IPs before TCP connect, blocks private/loopback/link-local/CGNAT/IPv6-mapped ranges, blocks non-80/443 ports and URL credentials, and validates every HTTP redirect hop manually.
  - **Signed Callbacks & Rate Limits**: HMAC-SHA256 user-bound inline buttons (`<= 64` bytes), per-user command & hourly LLM rate limits, atomic `DAILY_ORG_LLM_BUDGET`, outbound `TelegramRateLimiter`, XML delimiter sanitization against prompt injection, log token redaction, and `0o600` DB/`.env` permissions.

## Quick Start (macOS)

1. **Create and activate a Python 3.12 virtual environment**:
   ```bash
   cd /path/to/tele-news
   /opt/homebrew/bin/python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. **Configure `agent/.env`**:
   Copy `.env.example` to `.env` and set:
   - `TELEGRAM_BOT_TOKEN` — token from `@BotFather`
   - `ORG_NAME` — organization display name (e.g., `Acme AI Lab`)
   - `TELEGRAM_ADMIN_IDS` — comma-separated numeric Telegram user IDs for admins (or leave blank with `ALLOW_SELF_REGISTER_FIRST_ADMIN=true` so the first `/start` DM user becomes admin)
   - `TELEGRAM_ALLOWED_USER_IDS` — optional pre-approved member IDs (colleagues can also join via `/admin invite`)
   - `LLM_PROVIDER` & `LLM_MODEL` — `codex` (local Codex CLI OAuth) or `openai` (with `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`)
   - `ENABLE_HF_DAILY_PAPERS`, `ENABLE_ARXIV_API`, `SEMANTIC_SCHOLAR_API_KEY`
   - `DAILY_ORG_LLM_BUDGET`, `RATE_LIMIT_COMMANDS_PER_MIN`, `RATE_LIMIT_LLM_PER_USER_HOUR`

3. **Onboarding 20+ Team Members**:
   - Admin starts the bot in a 1:1 DM and runs `/admin invite member 25 72` (creates a 25-use invite valid for 72 hours, or generate 1-use invites per colleague).
   - Each colleague opens a 1:1 DM with the bot and sends `/join <invite_code>`, then votes on the 8-topic onboarding poll.
   - Admins can audit activity and manage roles anytime with `/admin users`, `/admin promote`, `/admin demote`, `/admin revoke`, `/admin audit`, and `/admin budget`.

4. **Run Tests & Start**:
   ```bash
   cd /path/to/tele-news
   .venv/bin/pytest -v
   .venv/bin/python main.py
   ```

## Command Reference

| Category | Commands |
| --- | --- |
| **Onboarding** | `/start`, `/join <code>`, `/help` |
| **Daily Digest** | `/digest [balanced\|papers\|news]`, `/topics`, `/poll`, `/discover`, `/settings`, `/pause`, `/resume` |
| **Literature Review** | `/review <ref>`, `/litreview <question>`, `/compare <id1> <id2> ...`, `/search [--arxiv] <query>`, `/trending` |
| **Library & Export** | `/library [status\|tag]`, `/save <ref> [tags]`, `/note <id> <text>`, `/read <id>`, `/bibtex <ref>`, `/export [bibtex\|markdown]` |
| **Admin & Ops** | `/feeds [add\|remove]`, `/status`, `/admin <invite\|users\|approve\|promote\|demote\|revoke\|audit\|budget>` |

## Architecture Decision Records

See [docs/adr/README.md](../docs/adr/README.md) for all ADRs (RSS feed validation, retention pruning, multi-user RBAC & security hardening, and literature review pipeline).
