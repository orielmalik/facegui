import httpx
from typing import Any, Dict, Optional


class ApiClient:

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient()

    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        response = await self.client.get(
            f"{self.base_url}{path}",
            params=params
        )
        response.raise_for_status()
        return response.json()

    async def post(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        response = await self.client.post(
            f"{self.base_url}{path}",
            json=data
        )
        response.raise_for_status()
        return response.json()

    async def put(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        response = await self.client.put(
            f"{self.base_url}{path}",
            json=data
        )
        response.raise_for_status()
        return response.json()

    async def patch(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        response = await self.client.patch(
            f"{self.base_url}{path}",
            json=data
        )
        response.raise_for_status()
        return response.json()

    async def delete(
        self,
        path: str
    ) -> Any:
        response = await self.client.delete(
            f"{self.base_url}{path}"
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()