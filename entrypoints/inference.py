#!/usr/bin/env python3
"""
vLLM inference server entrypoint for RLVR experiments.
"""
import argparse
import time
import uvicorn
import tomllib

from fastapi import FastAPI, Request
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import router as openai_router


import logging
logging.basicConfig(level=logging.DEBUG)


def create_app(vllm_config: dict) -> FastAPI:
    
    # create FastAPI app
    app = FastAPI(
        title="rlvr server",
        description="vllm inference engine",
        version="0.1.0",
    )
    
    # uptime tracking
    app.state.start_time = time.time()
    
    # Create engine arguments
    print("vllm config", vllm_config)
    engine_args = AsyncEngineArgs(**vllm_config)
    
    # intitiate the engine
    print(f"Initializing vLLM engine with model: {vllm_config.get('model')}")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    app.state.engine = engine
    app.state.model_name = vllm_config.get("model")
    
    # openAI-compatible routes
    app.include_router(openai_router, prefix="/v1")
    
    # Add custom RLVR endpoints
    @app.get("/health")
    async def health_check():
        """Basic health check."""
        return {
            "status": "healthy",
            "model": app.state.model_name,
            "uptime_seconds": time.time() - app.state.start_time,
        }
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        
        print(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
        return response
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Start vLLM inference server")
    parser.add_argument("config", type=str, help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, 'rb') as f:
        config = tomllib.load(f)
            
    print(f"Starting RLVR inference server")
    
    app = create_app(config["vllm"])
    
    uvicorn.run(app, **config["uvicorn"])


if __name__ == "__main__":
    main()