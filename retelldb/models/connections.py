"""
Connections: the identity anchors (anne, oct 29 2026).

the problem this answers: prixie meets the same person on zoom, then on
discord, then in a whatsapp call, then by email. without an anchor each
of those is a STRANGER: a brand new person she is joining on. a
connection is the pre-defined fact that makes them one person, with the
per-platform handles recorded once (discord username AND id, email,
whatsapp number, telegram, slack, imessage, meeting emails) so a scope
can resolve "who is in front of me" from whatever platform's handle it
holds.

two kinds of connection:
- the owner (owner=True): her own accounts, one per platform. knowing
  which account is HERS is what stops prixie from treating her own
  messages as a stranger's.
- everyone else: the people she interacts with.

honest rules:
- handles are stored normalized (emails and usernames lowercase, phones
  digits with a leading +, ids verbatim) so one person matches on every
  surface.
- a handle that matches two connections is AMBIGUOUS and resolves to
  none: guessing which of two people is in front of her is worse than
  not knowing (the same rule as host reading and personas).
- no real handles in the repo. deploys load them from env or the
  context store; the examples here use example.com placeholder shapes.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


# the surfaces prixie joins or answers on. a platform missing from here
# is UNKNOWN, and scope treats unknown as drop (fail closed).
PLATFORMS = (
    "zoom",
    "google_meet",
    "teams",
    "discord",
    "slack",
    "whatsapp",
    "imessage",
    "telegram",
    "email",
    "phone",
    "twitch",
    "kick",
    "youtube",
    "website",
)

# platform classes are what carry/drop decisions read. the class decides
# how private a surface is, not which brand it is.
PRIVATE_PLATFORMS = ("discord", "slack", "whatsapp", "imessage", "telegram", "email", "phone")
MEETING_PLATFORMS = ("zoom", "google_meet", "teams")
PUBLIC_SOCIAL_PLATFORMS = ("twitch", "kick", "youtube")
WEBSITE_PLATFORMS = ("website",)


def platform_class(platform: Optional[str]) -> str:
    """private | meeting | public_social | website | unknown."""
    p = (platform or "").strip().lower()
    if p in PRIVATE_PLATFORMS:
        return "private"
    if p in MEETING_PLATFORMS:
        return "meeting"
    if p in PUBLIC_SOCIAL_PLATFORMS:
        return "public_social"
    if p in WEBSITE_PLATFORMS:
        return "website"
    return "unknown"


def _e164ish(handle: str) -> str:
    digits = "".join(ch for ch in handle if ch.isdigit() or ch == "+")
    if not digits:
        return ""
    if digits.startswith("+"):
        return digits
    return "+" + digits


def normalize_handle(platform: str, handle: str) -> str:
    """One person, one normalized handle, whatever surface spelled it."""
    p = (platform or "").strip().lower()
    h = (handle or "").strip()
    if not h:
        return ""
    if p in ("email", "zoom", "google_meet", "teams"):
        # identities on meeting platforms are emails
        return h.lower()
    if p in ("whatsapp", "imessage", "telegram", "phone"):
        return _e164ish(h)
    # discord usernames are case-insensitive, ids are digits, slack ids
    # are opaque: lowercase is safe for all three, verbatim digits stay
    return h.lower()


@dataclass
class Connection:
    """One person, with the handles they hold on each platform."""

    id: str
    label: str
    owner: bool = False
    handles: Dict[str, List[str]] = field(default_factory=dict)
    notes: str = ""

    def matches(self, platform: str, handle: str) -> bool:
        p = (platform or "").strip().lower()
        wanted = normalize_handle(p, handle)
        if not wanted:
            return False
        return any(normalize_handle(p, h) == wanted for h in self.handles.get(p, []))

    def handles_for(self, platform: str) -> List[str]:
        return list(self.handles.get((platform or "").strip().lower(), []))


def resolve_connection(
    connections: Iterable[Connection],
    platform: str,
    handle: str,
) -> Optional[Connection]:
    """The one connection a handle belongs to, or None (none or ambiguous)."""
    hits = [c for c in connections if c.matches(platform, handle)]
    if len(hits) == 1:
        return hits[0]
    # zero is honest ignorance; more than one is ambiguity, both are None
    return None


def owner_connection(connections: Iterable[Connection]) -> Optional[Connection]:
    owners = [c for c in connections if c.owner]
    return owners[0] if len(owners) == 1 else None


def example_connections() -> List[Connection]:
    """Placeholder shapes only. real handles live in env or the store."""
    return [
        Connection(
            id="owner",
            label="the owner",
            owner=True,
            handles={
                "discord": ["owner-id-placeholder"],
                "email": ["owner@example.com"],
                "whatsapp": ["+15550000000"],
            },
        ),
        Connection(
            id="priya",
            label="priya",
            handles={
                "discord": ["priya-id-placeholder"],
                "email": ["priya@example.com"],
                "zoom": ["priya@example.com"],
                "google_meet": ["priya@example.com"],
            },
            notes="example counterpart: the same person on three surfaces",
        ),
    ]
