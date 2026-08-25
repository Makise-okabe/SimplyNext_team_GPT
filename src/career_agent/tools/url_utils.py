from urllib.parse import urlparse


def domain(url: str) -> str:
    return urlparse(url).netloc.lower()
