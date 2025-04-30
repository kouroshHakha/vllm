import argparse
import httpx
import logging
import uuid
from fastapi import FastAPI, BackgroundTasks
import uvicorn
from pydantic import BaseModel
from typing import Optional, Dict
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("proxy")

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int
    stream: bool = False

class ProxyService:
    def __init__(self, ip: str, port: int, prefill_url: str, decode_url: str):
        self.ip = ip
        self.port = port
        self.prefill_url = prefill_url
        self.decode_url = decode_url
        
        # Initialize FastAPI
        self.app = FastAPI()
        self.setup_routes()
        
        # HTTP client for sending requests
        self.client = httpx.AsyncClient()
    
    def setup_routes(self):
        @self.app.post("/generate")
        async def generate(request: GenerationRequest):
            # Generate a unique request_id
            request_id = str(uuid.uuid4())
            logger.info(f"Received generation request with assigned request_id: {request_id}")
            
            # Send requests to both prefill and decode services
            response = await self.send_requests(
                request_id=request_id,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                stream=request.stream
            )
            
            return response
    
    async def send_requests(self, request_id: str, prompt: str, max_tokens: int, stream: bool):
        """Send requests to both prefill and decode services"""
        try:
            # Schedule both requests concurrently
            prefill_task = asyncio.create_task(
                self.client.post(
                    f"{self.prefill_url}/prefill",
                    json={
                        "request_id": request_id,
                        "prompt": prompt,
                        "max_tokens": 1  # Just compute KV cache
                    }
                )
            )
            decode_task = asyncio.create_task(
                self.client.post(
                    f"{self.decode_url}/decode",
                    json={
                        "request_id": request_id,
                        "max_tokens": max_tokens
                    }
                )
            )

            # Await both tasks
            prefill_response, decode_response = await asyncio.gather(prefill_task, decode_task)

            logger.info(f"Prefill response: {prefill_response.json()}")
            logger.info(f"Decode response: {decode_response.json()}")

            return decode_response.json()

        except Exception as e:
            logger.error(f"Error sending requests: {e}")
    
    def start(self):
        """Start the FastAPI server"""
        uvicorn.run(self.app, host=self.ip, port=self.port, log_level="info")

def parse_args():
    parser = argparse.ArgumentParser(description="Proxy service for P2P KV cache transfer system")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="IP address to bind")
    parser.add_argument("--port", type=int, default=3000, help="Port to bind")
    parser.add_argument("--prefill-url", type=str, default="http://localhost:2000", help="Prefill service URL")
    parser.add_argument("--decode-url", type=str, default="http://localhost:2100", help="Decode service URL")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    service = ProxyService(
        ip=args.ip,
        port=args.port,
        prefill_url=args.prefill_url,
        decode_url=args.decode_url
    )
    service.start()
