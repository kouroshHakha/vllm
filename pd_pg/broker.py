import zmq
import json
import argparse
from typing import Dict, List, Set, Tuple
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("zmq-broker")

class ZMQBroker:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        
        # Dictionary to store producers and consumers by request_id
        # {request_id: {'producers': [{'rank': rank, 'ip': ip, 'port': port}, ...], 
        #               'consumers': [{'rank': rank, 'ip': ip, 'port': port}, ...]}}
        self.matches: Dict[str, Dict[str, List[Dict]]] = {}
        
        # Set to track completed matches
        self.completed_matches: Set[str] = set()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Initialize ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{ip}:{port}")
        
        logger.info(f"ZMQ Broker started on {ip}:{port}")
    
    def start(self):
        """Start the broker service loop"""
        while True:
            try:
                # Wait for a message from a client
                message = self.socket.recv_json()
                logger.info(f"Received message: {message}")
                
                # Process the message and send a response
                response = self.process_message(message)
                self.socket.send_json(response)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.socket.send_json({"status": "error", "message": str(e)})
    
    def process_message(self, message: Dict) -> Dict:
        """Process incoming messages from producers and consumers"""
        with self.lock:
            message_type = message.get("type")
            request_id = message.get("request_id")
            
            if not request_id:
                return {"status": "error", "message": "Missing request_id"}
            
            if message_type == "register_producer":
                return self._register_producer(request_id, message)
            elif message_type == "register_consumer":
                return self._register_consumer(request_id, message)
            elif message_type == "check_match":
                return self._check_match(request_id, message)
            else:
                return {"status": "error", "message": f"Unknown message type: {message_type}"}
    
    def _register_producer(self, request_id: str, message: Dict) -> Dict:
        """Register a producer for a request_id"""
        rank = message.get("rank")
        ip = message.get("ip")
        port = message.get("port")
        
        if not all([rank is not None, ip, port]):
            return {"status": "error", "message": "Missing required producer info"}
        
        if request_id not in self.matches:
            self.matches[request_id] = {"producers": [], "consumers": []}
            
        # Add producer info
        producer_info = {"rank": rank, "ip": ip, "port": port}
        self.matches[request_id]["producers"].append(producer_info)
        
        logger.info(f"Registered producer for request_id {request_id}, rank {rank} at {ip}:{port}")
        
        # Check if match is complete
        self._check_complete_match(request_id)
        
        return {"status": "success", "message": "Producer registered successfully"}
    
    def _register_consumer(self, request_id: str, message: Dict) -> Dict:
        """Register a consumer for a request_id"""
        rank = message.get("rank")
        ip = message.get("ip")
        port = message.get("port")
        
        if not all([rank is not None, ip, port]):
            return {"status": "error", "message": "Missing required consumer info"}
        
        if request_id not in self.matches:
            self.matches[request_id] = {"producers": [], "consumers": []}
            
        # Add consumer info
        consumer_info = {"rank": rank, "ip": ip, "port": port}
        self.matches[request_id]["consumers"].append(consumer_info)
        
        logger.info(f"Registered consumer for request_id {request_id}, rank {rank} at {ip}:{port}")
        
        # Check if match is complete
        self._check_complete_match(request_id)
        
        return {"status": "success", "message": "Consumer registered successfully"}
    
    def _check_match(self, request_id: str, message: Dict) -> Dict:
        """Check if a match exists for a request_id"""
        role = message.get("role")  # 'producer' or 'consumer'
        rank = message.get("rank")
        
        if not role or rank is None:
            return {"status": "error", "message": "Missing role or rank"}
        
        if request_id not in self.matches:
            return {"status": "pending", "message": "No match found yet"}
        
        if request_id in self.completed_matches:
            # Get the counterpart info
            counterpart_role = "consumers" if role == "producer" else "producers"
            counterparts = self.matches[request_id][counterpart_role]
            
            return {
                "status": "matched",
                "counterparts": counterparts,
                "message": f"Match found with {len(counterparts)} {counterpart_role}"
            }
        
        return {"status": "pending", "message": "Match is not complete yet"}
    
    def _check_complete_match(self, request_id: str) -> None:
        """Check if both producers and consumers are registered for a request_id"""
        if request_id in self.completed_matches:
            return
        
        match_data = self.matches.get(request_id, {})
        producers = match_data.get("producers", [])
        consumers = match_data.get("consumers", [])
        
        # Logic to determine if match is complete
        # This is a simplified version - in reality you might want to ensure same number of ranks, etc.
        if producers and consumers:
            self.completed_matches.add(request_id)
            logger.info(f"Completed match for request_id {request_id}")

def parse_args():
    parser = argparse.ArgumentParser(description="ZMQ Broker for matching prefill and decode instances")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="IP address to bind")
    parser.add_argument("--port", type=int, default=2500, help="Port to bind")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    broker = ZMQBroker(args.ip, args.port)
    broker.start()
