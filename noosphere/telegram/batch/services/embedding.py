"""Embedding service module for telexp."""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import httpx
import logging
import numpy as np

class EmbeddingModel(BaseModel):
    """Configuration for a specific embedding model."""
    name: str
    model: str
    vector_size: int

class EmbeddingService(BaseModel):
    """Configuration for an embedding service."""
    name: str
    type: str
    url: str
    models: List[EmbeddingModel]

class EmbeddingServiceManager:
    """Manager for embedding services."""
    
    def __init__(self, services: List[EmbeddingService]):
        """Initialize with list of service configurations."""
        self.services = {service.name: service for service in services}
        self.clients = {}
        self._setup_called = False
        self.logger_name = f"{self.__class__.__name__}_{id(self)}"
        
    def __getstate__(self):
        """Return state for pickling."""
        state = {
            'services': list(self.services.values()),
            'logger_name': self.logger_name
        }
        return state
        
    def __setstate__(self, state):
        """Restore state from pickle."""
        self.services = {service.name: service for service in state['services']}
        self.clients = {}
        self._setup_called = False
        self.logger_name = state['logger_name']

    def setup(self):
        """Initialize the manager."""
        if self._setup_called:
            return

        # Create logger
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.INFO)
        self._setup_called = True
        
    def teardown(self):
        """Clean up resources."""
        self.close_all()
        self._setup_called = False
        
    def get_service(self, name: str) -> Optional[EmbeddingService]:
        """Get service configuration by name."""
        if not self._setup_called:
            self.setup()
        return self.services.get(name)
        
    def get_model(self, service_name: str, model_name: str) -> Optional[EmbeddingModel]:
        """Get model configuration by service and model name.
        
        This will match against either the name field or the model field,
        allowing for more flexibility in configuration.
        """
        if not self._setup_called:
            self.setup()
        service = self.get_service(service_name)
        if not service:
            return None
        
        # First try to match by name field
        model = next((m for m in service.models if m.name == model_name), None)
        if model:
            return model
            
        # If not found, try to match by model field
        return next((m for m in service.models if m.model == model_name), None)
    
    def get_client(self, service_name: str, timeout: float = 30.0) -> Optional[httpx.Client]:
        """Get or create HTTP client for a service."""
        if not self._setup_called:
            self.setup()
        if service_name not in self.clients:
            service = self.get_service(service_name)
            if not service:
                return None
            self.clients[service_name] = httpx.Client(
                base_url=service.url,
                timeout=timeout
            )
        return self.clients[service_name]
    
    def close_all(self):
        """Close all HTTP clients."""
        for client in self.clients.values():
            client.close()
        self.clients.clear()

    def embed(
        self,
        service_name: str,
        model_name: str,
        text: str,
        timeout: float = 30.0
    ) -> Optional[List[float]]:
        """Generate embeddings using specified service and model."""
        if not self._setup_called:
            self.setup()
            
        service = self.get_service(service_name)
        model = self.get_model(service_name, model_name)
        client = self.get_client(service_name, timeout)
        
        if not all([service, model, client]):
            self.logger.error(
                f"Failed to get service configuration: "
                f"service={service_name}, model={model_name}"
            )
            return None
            
        try:
            # Make request to Ollama API
            response = client.post(
                "/api/embeddings",
                json={
                    "model": model.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Get embedding and validate size
            embedding = data.get("embedding")
            if not embedding or len(embedding) != model.vector_size:
                self.logger.error(
                    f"Invalid embedding size: got {len(embedding) if embedding else 0}, "
                    f"expected {model.vector_size}"
                )
                return None
                
            return embedding
            
        except Exception as e:
            self.logger.error(
                f"Failed to generate embedding: service={service_name}, "
                f"model={model_name}, error={str(e)}"
            )
            return None 