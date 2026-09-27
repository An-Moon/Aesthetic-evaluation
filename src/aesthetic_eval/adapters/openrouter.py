import base64
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from PIL import Image

from aesthetic_eval.adapters.base import BaseAdapter
from aesthetic_eval.data import EvalSample, load_and_resize_rgb


class OpenRouterAdapter(BaseAdapter):
    def __init__(self, base_cfg: dict, model_cfg: dict):
        super().__init__(base_cfg, model_cfg)
        self.api_key = ""
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout = 120
        self.retry_count = 3
        self.retry_sleep_seconds = 2.0
        self.request_concurrency = 1
        self.opener = urllib.request.build_opener()
        self.style_instruction = ""
        self.few_shot_examples: List[Dict[str, str]] = []

    def load(self) -> None:
        key_env = str(self.model_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
        self.api_key = os.environ.get(key_env, "").strip()
        if not self.api_key:
            raise RuntimeError(f"Missing OpenRouter API key: set {key_env}")

        self.api_url = str(self.model_cfg.get("api_url", self.api_url))
        self.timeout = int(self.model_cfg.get("timeout_seconds", 120))
        self.retry_count = int(self.model_cfg.get("retry_count", 3))
        self.retry_sleep_seconds = float(self.model_cfg.get("retry_sleep_seconds", 2.0))
        self.request_concurrency = max(1, int(self.model_cfg.get("request_concurrency", 1)))
        proxy_env = str(self.model_cfg.get("proxy_env", "OPENROUTER_PROXY"))
        proxy_url = str(
            self.model_cfg.get("proxy_url", "")
            or os.environ.get(proxy_env, "")
            or os.environ.get("HTTPS_PROXY", "")
            or os.environ.get("HTTP_PROXY", "")
            or os.environ.get("https_proxy", "")
            or os.environ.get("http_proxy", "")
        ).strip()
        self.model_cfg["proxy_active"] = bool(proxy_url)
        self.model_cfg["proxy_source"] = proxy_env if os.environ.get(proxy_env, "") else ("proxy_url" if proxy_url else "")
        if proxy_url:
            self.opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
            )
        self._load_prompt_style()

    def _resolve_style_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        repo_root = Path(__file__).resolve().parents[3]
        return repo_root / path

    def _load_prompt_style(self) -> None:
        style_cfg: Dict[str, Any] = {}
        style_path = str(self.model_cfg.get("prompt_style_file", "")).strip()
        if style_path:
            with self._resolve_style_path(style_path).open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Prompt style config must be a dict: {style_path}")
            style_cfg.update(loaded)

        style_cfg.update(self.model_cfg.get("prompt_style", {}) or {})
        self.style_instruction = str(style_cfg.get("instruction", "")).strip()
        examples = style_cfg.get("examples", []) or []
        if not isinstance(examples, list):
            raise ValueError("prompt_style.examples must be a list")
        self.few_shot_examples = []
        for item in examples:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            reference = str(item.get("reference", "")).strip()
            if question and reference:
                self.few_shot_examples.append(
                    {
                        "id": str(item.get("id", "")).strip(),
                        "image": str(item.get("image", "")).strip(),
                        "question": question,
                        "reference": reference,
                    }
                )

    def build_prompt(self, sample: EvalSample) -> str:
        if self._text_context:
            return self.format_prompt(sample, "Please describe the aesthetic quality of this image.")
        template = str(self.base_cfg.get("prompt", {}).get("template", "{question}"))
        question = template.format(question=sample.question)
        if not self.style_instruction and not self.few_shot_examples:
            return question

        parts = []
        if self.style_instruction:
            parts.append(self.style_instruction)

        examples = []
        for item in self.few_shot_examples:
            same_id = bool(item.get("id")) and item["id"] == sample.sample_id
            same_image = bool(item.get("image")) and item["image"] == sample.image
            if same_id or same_image:
                continue
            examples.append(
                "Question: {question}\nReference answer: {reference}".format(
                    question=item["question"],
                    reference=item["reference"],
                )
            )

        if examples:
            parts.append("Reference answer style examples:\n\n" + "\n\n".join(examples))

        parts.append(
            "Now answer the current image question in the same reference-answer style.\n"
            f"Question: {question}\nAnswer:"
        )
        return "\n\n".join(parts)

    def _image_to_data_url(self, image: Image.Image) -> str:
        quality = int(self.model_cfg.get("image_jpeg_quality", 90))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def prepare_batch(self, batch_samples: List[EvalSample]) -> Tuple[Any, List[EvalSample], List[str]]:
        image_size = int(self.base_cfg.get("preprocess", {}).get("image_size", 448))

        prepared = []
        valid: List[EvalSample] = []
        prompts: List[str] = []

        for sample in batch_samples:
            try:
                image = load_and_resize_rgb(sample.image_resolved, image_size)
                prompt = self.build_prompt(sample)
                prepared.append(
                    {
                        "prompt": prompt,
                        "image_data_url": self._image_to_data_url(image),
                    }
                )
                valid.append(sample)
                prompts.append(prompt)
            except Exception:
                continue

        if not valid:
            return None, [], []
        return prepared, valid, prompts

    def _payload(self, item: Dict[str, str]) -> Dict[str, Any]:
        gen_cfg = self.base_cfg.get("generation", {})
        payload: Dict[str, Any] = {
            "model": self.model_cfg["openrouter_model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item["prompt"]},
                        {"type": "image_url", "image_url": {"url": item["image_data_url"]}},
                    ],
                }
            ],
            "max_tokens": int(gen_cfg.get("max_new_tokens", 256)),
            "stream": False,
        }

        if gen_cfg.get("temperature") is not None:
            payload["temperature"] = float(gen_cfg["temperature"])
        elif not bool(gen_cfg.get("do_sample", False)):
            payload["temperature"] = 0.0
        if gen_cfg.get("top_p") is not None:
            payload["top_p"] = float(gen_cfg["top_p"])
        if gen_cfg.get("top_k") is not None:
            payload["top_k"] = int(gen_cfg["top_k"])
        if self.model_cfg.get("reasoning") is not None:
            payload["reasoning"] = self.model_cfg["reasoning"]
        if self.model_cfg.get("provider") is not None:
            payload["provider"] = self.model_cfg["provider"]
        return payload

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
        }
        referer = self.model_cfg.get("http_referer")
        title = self.model_cfg.get("x_title", "aesthetic_eval_framework")
        if referer:
            headers["HTTP-Referer"] = str(referer)
        if title:
            headers["X-Title"] = str(title)
        return headers

    def _request_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with self.opener.open(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _request_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error = ""
        for attempt in range(self.retry_count + 1):
            try:
                return self._request_once(payload)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {body[:500]}"
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt >= self.retry_count:
                    break
                retry_after = exc.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else self.retry_sleep_seconds
                time.sleep(sleep_s)
            except Exception as exc:
                last_error = str(exc)
                if attempt >= self.retry_count:
                    break
                time.sleep(self.retry_sleep_seconds)
        raise RuntimeError(last_error)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text", item.get("content", ""))
                    if text:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return "".join(parts).strip()
        return "" if content is None else str(content).strip()

    def _call_one(self, item: Dict[str, str]) -> str:
        try:
            response = self._request_with_retry(self._payload(item))
            choice = response.get("choices", [{}])[0]
            if choice.get("error"):
                raise RuntimeError(f"OpenRouter choice error: {choice['error']}")
            message = choice.get("message", {})
            text = self._content_to_text(message.get("content"))
            if not text:
                raise RuntimeError(f"OpenRouter returned empty content: {response}")
            return text
        except Exception:
            raise

    def generate_batch(self, prepared: Any) -> List[str]:
        items = list(prepared)
        workers = min(self.request_concurrency, len(items))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self._call_one, items))
