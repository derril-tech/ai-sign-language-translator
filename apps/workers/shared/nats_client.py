import asyncio
import json
import logging
from typing import Any, Callable, Optional
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
import os

logger = logging.getLogger(__name__)

class NATSClient:
    def __init__(self, servers: Optional[str] = None):
        self.servers = servers or os.getenv("NATS_URL", "nats://localhost:4222")
        self.nc: Optional[NATS] = None
        
    async def connect(self):
        """Connect to NATS server"""
        self.nc = NATS()
        await self.nc.connect(self.servers)
        logger.info(f"Connected to NATS at {self.servers}")
        
    async def disconnect(self):
        """Disconnect from NATS server"""
        if self.nc:
            await self.nc.close()
            logger.info("Disconnected from NATS")
            
    async def publish(self, subject: str, data: Any):
        """Publish message to NATS subject"""
        if not self.nc:
            raise RuntimeError("Not connected to NATS")
            
        message = json.dumps(data).encode()
        await self.nc.publish(subject, message)
        logger.debug(f"Published to {subject}: {data}")
        
    async def subscribe(self, subject: str, handler: Callable[[Any], None], queue: Optional[str] = None):
        """Subscribe to NATS subject with handler"""
        if not self.nc:
            raise RuntimeError("Not connected to NATS")
            
        async def message_handler(msg: Msg):
            try:
                data = json.loads(msg.data.decode())
                await handler(data)
            except Exception as e:
                logger.error(f"Error handling message from {subject}: {e}")
                
        await self.nc.subscribe(subject, cb=message_handler, queue=queue)
        logger.info(f"Subscribed to {subject}" + (f" (queue: {queue})" if queue else ""))
        
    async def request(self, subject: str, data: Any, timeout: float = 5.0) -> Any:
        """Send request and wait for response"""
        if not self.nc:
            raise RuntimeError("Not connected to NATS")
            
        message = json.dumps(data).encode()
        response = await self.nc.request(subject, message, timeout=timeout)
        return json.loads(response.data.decode())
