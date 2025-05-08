import argparse
import json
import logging
import threading
import time
from typing import Dict, List
import uuid
import zmq.asyncio
from fastapi import FastAPI, BackgroundTasks
import uvicorn
from pydantic import BaseModel
import asyncio
from proxy import GenerationRequest


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("prefill")


class PrefillWorker:
    def __init__(self, rank: int, ip: str, port: int, zmq_ip: str, zmq_port: int):
        self.rank = rank
        self.ip = ip
        self.port = port  # Base port + rank offset
        self.zmq_ip = zmq_ip
        self.zmq_port = zmq_port
        
        # Use the asyncio-based ZMQ context & socket
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{zmq_ip}:{zmq_port}")
        
        logger.info(f"Prefill worker rank {rank} initialized at {ip}:{port}")
        
        # Async lock for pending_requests
        self.pending_requests: Dict[str, Dict] = {}
        self.lock = asyncio.Lock()
    
    async def register_with_broker(self, request_id: str) -> Dict:
        """
        Register this worker as a producer for the given request_id
        using the asyncio-enabled ZMQ socket.
        """
        message = {
            "type": "register_producer",
            "request_id": request_id,
            "rank": self.rank,
            "ip": self.ip,
            "port": self.port,
        }
        
        logger.info(f"Rank {self.rank} registering as producer for request_id {request_id}")
        
        # These are now coroutines you can await
        await self.socket.send_json(message)
        reply = await self.socket.recv_json()
        return reply
    
    async def check_match(self, request_id: str) -> Dict:
        """Check if there is a consumer match for this request_id"""
        message = {
            "type": "check_match",
            "request_id": request_id,
            "role": "producer",
            "rank": self.rank
        }
        
        await self.socket.send_json(message)
        return await self.socket.recv_json()
    
    async def process_request(self, request_id: str, prompt: str, max_tokens: int):
        """Process a prefill request"""
        # Properly acquire the asyncio.Lock
        async with self.lock:
            self.pending_requests[request_id] = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "status": "processing"
            }
        
        result = await self.register_with_broker(request_id)
        logger.info(f"Rank {self.rank} registration result: {result}")
        
        # Wait for consumer match
        matched = False
        consumers = []
        
        while not matched:
            match_result = await self.check_match(request_id)
            if match_result["status"] == "matched":
                matched = True
                consumers = match_result["counterparts"]
                logger.info(f"Rank {self.rank} matched with consumers for request_id {request_id}: {consumers}")
            else:
                logger.info(f"Rank {self.rank} waiting for match: {match_result}")
                await asyncio.sleep(1)
        
        # Simulate sending KV cache to decode instance
        # In a real implementation, you would establish NCCL connections here
        logger.info(f"Rank {self.rank} would now establish NCCL connection with decode rank {self.rank}")
        logger.info(f"Rank {self.rank} would transfer KV cache to decode rank {self.rank}")
        
        # Simulate computation time
        logger.info(f"Rank {self.rank} starting prefill computation for request_id {request_id}")
        # Simulate prefill computation
        await asyncio.sleep(0.2)
        
        # Update request status
        async with self.lock:
            if request_id in self.pending_requests:
                self.pending_requests[request_id]["status"] = "completed"

class PrefillService:
    def __init__(self, ip: str, port: int, nranks: int, zmq_ip: str, zmq_port: int):
        self.ip = ip
        self.port = port
        self.nranks = nranks
        self.zmq_ip = zmq_ip
        self.zmq_port = zmq_port
        
        # Initialize workers for each rank
        self.workers = []
        for rank in range(nranks):
            worker = PrefillWorker(
                rank=rank,
                ip=ip,
                port=port + rank + 1,  # Use port+1, port+2, etc. for worker ZMQ ports
                zmq_ip=zmq_ip,
                zmq_port=zmq_port
            )
            self.workers.append(worker)
        
        # Initialize FastAPI
        self.app = FastAPI()
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.post("/generate")
        async def generate(request: GenerationRequest):
            # Generate a request_id if not provided
            if not request.request_id:
                request.request_id = str(uuid.uuid4())
            
            logger.info(f"Received prefill request: {request}")
            
            # Start processing on all ranks in the background
            worker_tasks = []
            for worker in self.workers:
                worker_tasks.append(
                    worker.process_request(
                        request_id=request.request_id,
                        prompt=request.prompt,
                        max_tokens=request.max_tokens
                    )
                )
            
            await asyncio.gather(*worker_tasks)
            
            return {
                "status": "success",
                "request_id": request.request_id,
                "message": f"Prefill request {request.request_id} completed."
            }
    
    def start(self):
        """Start the FastAPI server"""
        uvicorn.run(self.app, host=self.ip, port=self.port, log_level="info")

def parse_args():
    parser = argparse.ArgumentParser(description="Prefill service for P2P KV cache transfer")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="IP address to bind")
    parser.add_argument("--port", type=int, default=2000, help="Port to bind")
    parser.add_argument("--nranks", type=int, default=4, help="Number of ranks")
    parser.add_argument("--zmq-ip", type=str, default="0.0.0.0", help="ZMQ broker IP")
    parser.add_argument("--zmq-port", type=int, default=2500, help="ZMQ broker port")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    service = PrefillService(
        ip=args.ip,
        port=args.port,
        nranks=args.nranks,
        zmq_ip=args.zmq_ip,
        zmq_port=args.zmq_port
    )
    service.start()
