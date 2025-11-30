import logging
import time
from typing import Any, Dict, List, Optional, Union, Iterator
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class ManifoldAPI:
    """
    Unified client for Manifold Markets API.
    Handles authentication, retries, pagination, and rate limiting.
    """
    
    BASE_URL = "https://api.manifold.markets/v0"
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "BetterManifoldBot/1.0",
            "Accept": "application/json"
        })
        
        if api_key:
            self.session.headers.update({"Authorization": f"Key {api_key}"})
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, requests.ConnectionError))
    )
    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make a request with automatic retries."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            
            # Raise for 4xx/5xx errors
            response.raise_for_status()
            
            return response.json()
            
        except requests.HTTPError as e:
            if e.response.status_code == 400:
                logger.error(f"Bad Request to {url}: {e.response.text}")
            raise

    def get_me(self) -> Dict[str, Any]:
        """Get current user info."""
        return self._request("GET", "me")

    def get_market(self, market_id: str) -> Dict[str, Any]:
        """Get a single market by ID or slug."""
        return self._request("GET", f"market/{market_id}")

    def get_markets(self, limit: int = 1000, before: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get a list of markets."""
        params = {"limit": limit}
        if before:
            params["before"] = before
        return self._request("GET", "markets", params=params)

    def get_all_markets_by_creator(self, username: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all markets created by a specific user.
        Since the API doesn't support filtering by creator directly in the /markets endpoint,
        we have to fetch markets and filter client-side.

        To be efficient, we scan backwards from the newest markets.
        """
        logger.info(f"Fetching markets for creator: {username}")

        found_markets = []
        before_cursor = None
        page_size = 1000
        max_pages = 10  # Stop after 10 pages (10,000 markets) to prevent infinite loops
        pages_scanned = 0

        while pages_scanned < max_pages:
            batch = self.get_markets(limit=page_size, before=before_cursor)
            if not batch:
                break

            pages_scanned += 1

            # Filter for the target user
            # We check both creatorUsername and creatorName to be safe
            user_batch = [
                m for m in batch
                if m.get('creatorUsername') == username or m.get('creatorName') == username
            ]

            found_markets.extend(user_batch)
            logger.info(f"Scanned {len(batch)} markets, found {len(user_batch)} by {username}. Total found: {len(found_markets)}")

            # Early termination if we have enough
            if limit and len(found_markets) >= limit:
                return found_markets[:limit]

            # Early termination: if no matches in this batch and we've scanned 3+ pages, likely won't find more
            if len(user_batch) == 0 and pages_scanned >= 3 and len(found_markets) > 0:
                logger.info(f"No matches in page {pages_scanned}, terminating search")
                break

            # Safety break
            if len(batch) < page_size:
                break

            # Update cursor
            before_cursor = batch[-1]['id']

        return found_markets

    def place_bet(self, market_id: str, outcome: str, amount: int) -> Dict[str, Any]:
        """Place a bet."""
        if not self.api_key:
            raise ValueError("API key required for betting")
            
        data = {
            "contractId": market_id,
            "outcome": outcome,
            "amount": amount
        }
        return self._request("POST", "bet", json=data)

    def cancel_bet(self, bet_id: str) -> Dict[str, Any]:
        """Cancel a bet."""
        if not self.api_key:
            raise ValueError("API key required for betting")
        return self._request("POST", f"bet/{bet_id}/cancel")
