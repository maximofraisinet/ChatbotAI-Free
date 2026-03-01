"""
Chat History Manager
Handles saving, loading, listing, and parsing Markdown chat files.
Auto-generates titles using the lightest available Ollama model.
"""

import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import ollama

CHATS_DIR = Path("chats")


# ── Utility: lightest Ollama model ────────────────────────────────────────────

def get_lightest_ollama_model() -> Optional[str]:
    """Return the name of the smallest (by bytes) locally-installed Ollama model,
    or None if no models are available."""
    try:
        response = ollama.list()
        models = response.models
        if not models:
            return None
        lightest = min(models, key=lambda m: getattr(m, "size", float("inf")))
        name = lightest.model
        print(f"⚡ Lightest Ollama model: {name} ({lightest.size / 1e6:.0f} MB)")
        return name
    except Exception as e:
        print(f"Error finding lightest model: {e}")
        return None


def generate_chat_title(user_message: str, model: Optional[str] = None) -> str:
    """Ask the lightest model for a ≤4-word title for *user_message*.
    Falls back to the first ~30 chars of the message on error."""
    if model is None:
        model = get_lightest_ollama_model()
    if model is None:
        return user_message[:30].strip()
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Generate a very short title (maximum 4 words) for this query. "
                        "Reply ONLY with the title, no quotes or explanations.\n\n"
                        f"{user_message}"
                    ),
                }
            ],
        )
        title = resp["message"]["content"].strip().strip('"').strip("'")
        # Limit to 60 chars just in case
        return title[:60] if title else user_message[:30].strip()
    except Exception as e:
        print(f"Title generation error: {e}")
        return user_message[:30].strip()


# ── Markdown I/O ──────────────────────────────────────────────────────────────

def _ensure_chats_dir():
    CHATS_DIR.mkdir(exist_ok=True)


def _sanitize_filename(name: str) -> str:
    """Remove characters that are unsafe in file names."""
    return re.sub(r'[\\/:*?"<>|]', "", name).strip()


def create_new_chat() -> dict:
    """Create a new empty chat metadata dict (no file written yet).

    Returns:
        dict with keys: id, title, filename, created, messages
    """
    chat_id = uuid.uuid4().hex[:12]
    now = datetime.now()
    return {
        "id": chat_id,
        "title": "New Chat",
        "filename": f"{now.strftime('%Y%m%d_%H%M%S')}_{chat_id}.md",
        "created": now,
        "messages": [],  # list of {"role": ..., "content": ...}
    }


def save_chat_to_md(chat: dict):
    """Write/overwrite a .md file from chat metadata.

    chat must have: filename, title, created, messages
    """
    _ensure_chats_dir()
    path = CHATS_DIR / chat["filename"]

    lines = [
        f"# Chat: {chat['title']}",
        f"*Date: {chat['created'].strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "---",
        "",
    ]
    for msg in chat["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append("### 👤 User")
        else:
            lines.append("### ✨ Bot")
        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def update_chat_title(chat: dict, new_title: str):
    """Update the title in the chat dict and re-save the file."""
    chat["title"] = new_title
    save_chat_to_md(chat)


def load_chat_from_md(filename: str) -> Optional[dict]:
    """Parse a .md chat file and return a chat dict, or None on error."""
    path = CHATS_DIR / filename
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    # Parse title
    title = "Untitled"
    title_match = re.search(r"^# Chat:\s*(.+)$", text, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    # Parse date
    created = datetime.now()
    date_match = re.search(r"\*Date:\s*([\d-]+ [\d:]+)\*", text)
    if date_match:
        try:
            created = datetime.strptime(date_match.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    # Parse message blocks
    messages: list[dict] = []
    # Split on "### 👤 User" or "### ✨ Bot"
    block_pattern = re.compile(
        r"###\s+(👤 User|✨ Bot)\s*\n(.*?)(?=\n###\s+(?:👤 User|✨ Bot)|\n?---\s*$|\Z)",
        re.DOTALL,
    )
    for m in block_pattern.finditer(text):
        role_str = m.group(1)
        content = m.group(2).strip().rstrip("-").strip()
        role = "user" if "User" in role_str else "assistant"
        if content:
            messages.append({"role": role, "content": content})

    chat_id = Path(filename).stem[-12:] or uuid.uuid4().hex[:12]
    return {
        "id": chat_id,
        "title": title,
        "filename": filename,
        "created": created,
        "messages": messages,
    }


def list_chats() -> list[dict]:
    """Return a list of chat metadata dicts, sorted newest-first.
    Only reads headers (title + date) for speed — messages are NOT loaded."""
    _ensure_chats_dir()
    chats = []
    for f in sorted(CHATS_DIR.glob("*.md"), reverse=True):
        try:
            # Read only first 3 lines for speed
            with open(f, encoding="utf-8") as fh:
                head = "".join(fh.readline() for _ in range(3))
        except Exception:
            continue

        title = "Untitled"
        title_match = re.search(r"^# Chat:\s*(.+)$", head, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()

        created = datetime.fromtimestamp(f.stat().st_mtime)
        date_match = re.search(r"\*Date:\s*([\d-]+ [\d:]+)\*", head)
        if date_match:
            try:
                created = datetime.strptime(date_match.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        chats.append({
            "id": f.stem[-12:],
            "title": title,
            "filename": f.name,
            "created": created,
        })
    return chats


def rename_chat(filename: str, new_title: str):
    """Load a chat, update its title, and re-save the .md file."""
    chat = load_chat_from_md(filename)
    if chat is None:
        return
    chat["title"] = new_title
    save_chat_to_md(chat)


def delete_chat(filename: str) -> bool:
    """Delete a chat .md file. Returns True on success."""
    path = CHATS_DIR / filename
    try:
        path.unlink(missing_ok=True)
        return True
    except Exception:
        return False
