import argparse
import json
import logging
import threading
import time
from typing import Dict, List
import uuid
import zmq.asyncio
import asyncio
from fastapi import FastAPI, BackgroundTasks
import uvicorn
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("decode")

class DecodeRequest(BaseModel):
    request_id: str
    max_tokens: int

class DecodeWorker:
    def __init__(self, rank: int, ip: str, port: int, zmq_ip: str, zmq_port: int):
        self.rank = rank
        self.ip = ip
        self.port = port  # Base port + rank offset
        self.zmq_ip = zmq_ip
        self.zmq_port = zmq_port
        
        # ZMQ client for communicating with the broker
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{zmq_ip}:{zmq_port}")
        
        logger.info(f"Decode worker rank {rank} initialized at {ip}:{port}")
        
        # Dict to store pending requests
        self.pending_requests = {}
        self.lock = asyncio.Lock()
    
    async def register_with_broker(self, request_id: str) -> Dict:
        """Register this worker as a consumer for the given request_id"""
        message = {
            "type": "register_consumer",
            "request_id": request_id,
            "rank": self.rank,
            "ip": self.ip,
            "port": self.port
        }
        
        logger.info(f"Rank {self.rank} registering as consumer for request_id {request_id}")
        await self.socket.send_json(message)
        return await self.socket.recv_json()
    
    async def check_match(self, request_id: str) -> Dict:
        """Check if there is a producer match for this request_id"""
        message = {
            "type": "check_match",
            "request_id": request_id,
            "role": "consumer",
            "rank": self.rank
        }
        
        await self.socket.send_json(message)
        return await self.socket.recv_json()
    
    async def process_request(self, request_id: str, max_tokens: int):
        """Process a decode request"""
        # Properly acquire the asyncio.Lock
        async with self.lock:
            # Store the request
            self.pending_requests[request_id] = {
                "max_tokens": max_tokens,
                "status": "processing"
            }
        
        # Register with broker
        result = await self.register_with_broker(request_id)
        logger.info(f"Rank {self.rank} registration result: {result}")
        
        # Wait for producer match
        matched = False
        producers = []
        
        while not matched:
            match_result = await self.check_match(request_id)
            if match_result["status"] == "matched":
                matched = True
                producers = match_result["counterparts"]
                logger.info(f"Rank {self.rank} matched with producers for request_id {request_id}: {producers}")
            else:
                logger.info(f"Rank {self.rank} waiting for match: {match_result}")
                await asyncio.sleep(1)
        
        # Simulate receiving KV cache from prefill instance
        # In a real implementation, you would establish NCCL connections here
        logger.info(f"Rank {self.rank} would now establish NCCL connection with prefill rank {self.rank}")
        logger.info(f"Rank {self.rank} would receive KV cache from prefill rank {self.rank}")
        
        # Simulate decoding
        logger.info(f"Rank {self.rank} starting decoding for request_id {request_id}")
        await asyncio.sleep(3)
        
        # Update request status
        async with self.lock:
            if request_id in self.pending_requests:
                self.pending_requests[request_id]["status"] = "completed"

class DecodeService:
    def __init__(self, ip: str, port: int, nranks: int, zmq_ip: str, zmq_port: int):
        self.ip = ip
        self.port = port
        self.nranks = nranks
        self.zmq_ip = zmq_ip
        self.zmq_port = zmq_port
        
        # Initialize workers for each rank
        self.workers = []
        for rank in range(nranks):
            worker = DecodeWorker(
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
        @self.app.post("/decode")
        async def decode(request: DecodeRequest):
            logger.info(f"Received decode request: {request}")
            
            # Start processing on all ranks in the background
            worker_tasks = []
            for worker in self.workers:
                worker_tasks.append(
                    worker.process_request(
                        request_id=request.request_id,
                        max_tokens=request.max_tokens
                    )
                )
            
            await asyncio.gather(*worker_tasks)
            
            return {
                "status": "success",
                "request_id": request.request_id,
                "message": f"Decode request {request.request_id} completed."
            }
    
    def start(self):
        """Start the FastAPI server"""
        uvicorn.run(self.app, host=self.ip, port=self.port, log_level="info")

def parse_args():
    parser = argparse.ArgumentParser(description="Decode service for P2P KV cache transfer")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="IP address to bind")
    parser.add_argument("--port", type=int, default=2100, help="Port to bind")
    parser.add_argument("--nranks", type=int, default=4, help="Number of ranks")
    parser.add_argument("--zmq-ip", type=str, default="0.0.0.0", help="ZMQ broker IP")
    parser.add_argument("--zmq-port", type=int, default=2500, help="ZMQ broker port")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    service = DecodeService(
        ip=args.ip,
        port=args.port,
        nranks=args.nranks,
        zmq_ip=args.zmq_ip,
        zmq_port=args.zmq_port
    )
    service.start()
