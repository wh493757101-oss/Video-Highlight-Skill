import base64
import json
import os
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image


@dataclass
class ArkConfig:
    api_key: str = field(default_factory=lambda: os.getenv("ARK_API_KEY", ""))
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    model: str = "doubao-seed-2-0-pro"
    max_retries: int = 3
    timeout: float = 120.0


def _encode_image(image_path: str) -> str:
    img = Image.open(image_path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    buf = BytesIO()
    ext = Path(image_path).suffix.lower()
    fmt = "JPEG" if ext in (".jpg", ".jpeg") else "PNG"
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


class ArkClient:
    def __init__(self, config: ArkConfig | None = None):
        self.config = config or ArkConfig()
        if not self.config.api_key:
            raise ValueError("ARK_API_KEY 未设置，请通过环境变量或 ArkConfig 提供")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        last_error: str | None = None
        for attempt in range(self.config.max_retries):
            try:
                resp = httpx.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                    timeout=self.config.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                if e.response.status_code == 429:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise
            except httpx.RequestError as e:
                last_error = str(e)
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)
                    continue
                raise

        raise RuntimeError(f"Ark API 调用失败（已重试 {self.config.max_retries} 次）: {last_error}")

    def chat_with_images(
        self,
        text: str,
        image_paths: list[str],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for path in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": _encode_image(path)},
            })

        messages = [{"role": "user", "content": content}]
        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def extract_json(self, response: dict[str, Any]) -> dict[str, Any]:
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return json.loads(content)
        return {}
