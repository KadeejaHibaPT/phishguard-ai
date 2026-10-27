import ipaddress
from urllib.parse import urlparse

URL_FEATURES = [
    "url_length",
    "hostname_length",
    "path_length",
    "count_dots",
    "count_hyphens",
    "count_at",
    "count_question",
    "count_percent",
    "count_equals",
    "count_slash",
    "count_digits",
    "count_subdomains",
    "has_https",
    "has_ip",
    "has_port",
    "has_punycode",
    "has_suspicious_word",
    "has_shortener",
    "has_double_slash_path",
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl",
    "ow.ly", "is.gd", "buff.ly", "cutt.ly",
}

SUSPICIOUS_WORDS = (
    "login", "verify", "account", "update", "secure",
    "bank", "signin", "password", "confirm", "wallet",
)


def extract_url_features(value: str) -> dict:
    """Extract features from a URL string."""
    raw = (value or "").strip()

    if not raw or any(char.isspace() for char in raw):
        raise ValueError("Enter a URL without spaces.")

    candidate = raw if "://" in raw else "https://" + raw
    parsed = urlparse(candidate)
    hostname = (parsed.hostname or "").lower().rstrip(".")

    if not hostname or ("." not in hostname and hostname != "localhost"):
        raise ValueError("Enter a valid-looking hostname, such as example.com.")

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    labels = hostname.split(".")
    text = raw.lower()

    return {
        "url_length": len(raw),
        "hostname_length": len(hostname),
        "path_length": len(parsed.path),
        "count_dots": raw.count("."),
        "count_hyphens": raw.count("-"),
        "count_at": raw.count("@"),
        "count_question": raw.count("?"),
        "count_percent": raw.count("%"),
        "count_equals": raw.count("="),
        "count_slash": raw.count("/"),
        "count_digits": sum(char.isdigit() for char in raw),
        "count_subdomains": max(0, len(labels) - 2),
        "has_https": int(parsed.scheme.lower() == "https"),
        "has_ip": has_ip,
        "has_port": int(parsed.port is not None),
        "has_punycode": int("xn--" in hostname),
        "has_suspicious_word": int(
            any(word in text for word in SUSPICIOUS_WORDS)
        ),
        "has_shortener": int(hostname in SHORTENERS),
        "has_double_slash_path": int("//" in parsed.path),
    }