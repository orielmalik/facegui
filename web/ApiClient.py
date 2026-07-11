import httpx
from typing import Any, Dict, Optional


class ApiClient:

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/") + "/"
        self.client = httpx.Client(base_url=self.base_url)

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        clean_path = path.lstrip("/")
        # שים לב: אין כאן await, זו קריאה סינכרונית רגילה
        response = self.client.request(
            method=method,
            url=clean_path,
            params=params,
            json=json_data,
        )
        response.raise_for_status()

        if response.status_code == 204 or not response.content:
            return None

        return response.json()

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, data: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("POST", path, json_data=data)

    def put(self, path: str, data: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("PUT", path, json_data=data)

    def patch(self, path: str, data: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("PATCH", path, json_data=data)

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    def close(self):
        self.client.close()