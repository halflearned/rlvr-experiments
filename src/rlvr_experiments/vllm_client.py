import logging
import socket
import time
from urllib.parse import urlparse

import requests
import torch

logger = logging.getLogger(__name__)


class VLLMClient:
    """Simple client for standard vLLM OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
    ):
        self.session = requests.Session()
        self.model_name = model
        
        if base_url is not None:
            parsed_url = urlparse(base_url)
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            self.host = host
            self.base_url = f"http://{self.host}:{server_port}"
        self.check_server(total_timeout=120)  # TODO: un-hardcode

    def check_server(self, total_timeout: float = 60.0, retry_interval: float = 5.0):
        """Check if server is available."""
        url = f"{self.base_url}/health"
        start_time = time.time()

        while True:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return
            except requests.exceptions.RequestException as exc:
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"Can't reach vLLM server at {self.base_url} after {total_timeout}s"
                    ) from exc
            
            logger.info(f"Server not ready. Retrying in {retry_interval}s...")
            time.sleep(retry_interval)

    def generate(
        self,
        prompts: str | list[str],
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 16,
        **kwargs,
    ) -> dict:
        """Generate completions using standard vLLM OpenAI API."""
        url = f"{self.base_url}/v1/completions"
        
        # Handle single prompt or list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        all_results = []
        for prompt in prompts:
            response = self.session.post(
                url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "n": n,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    **kwargs,
                },
                timeout=120,
            )
            
            if response.status_code == 200:
                all_results.append(response.json())
            else:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")
        
        return all_results
