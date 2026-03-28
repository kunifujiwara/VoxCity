# Utility functions for downloading files from various sources
import time
import requests
from ..utils.logging import get_logger

_logger = get_logger(__name__)


def download_file(url, filename, *, timeout=60, max_retries=3, initial_delay=2.0,
                  backoff_factor=2.0, chunk_size=8192):
    """Download a file from a URL and save it locally with retry and streaming.

    Uses streaming to avoid loading large files entirely into memory and retries
    on transient network failures with exponential backoff.

    Args:
        url (str): URL of the file to download.
        filename (str): Local path where the downloaded file will be saved.
        timeout (int): Request timeout in seconds (default 60).
        max_retries (int): Number of retry attempts on failure (default 3).
        initial_delay (float): Seconds to wait before the first retry (default 2.0).
        backoff_factor (float): Multiplier for delay between retries (default 2.0).
        chunk_size (int): Bytes per chunk when streaming (default 8192).

    Raises:
        requests.HTTPError: If download fails after all retries.

    Example:
        >>> download_file('https://example.com/file.pdf', 'local_file.pdf')
    """
    last_error = None
    for attempt in range(max_retries):
        if attempt > 0:
            delay = initial_delay * (backoff_factor ** (attempt - 1))
            _logger.info("Retry %d/%d: waiting %.1fs...", attempt, max_retries - 1, delay)
            time.sleep(delay)
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(filename, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        file.write(chunk)
            _logger.info("File downloaded successfully and saved as %s", filename)
            return
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            _logger.warning("Download attempt %d failed: %s", attempt + 1, exc)

    raise requests.HTTPError(
        f"Failed to download {url} after {max_retries} attempts: {last_error}"
    )