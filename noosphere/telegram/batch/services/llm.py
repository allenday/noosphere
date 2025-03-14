"""LLM service module for telexp."""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import httpx
import logging

class LLMModel(BaseModel):
    """Configuration for a specific LLM model."""
    name: str
    model: str
    prompt: str = None
    default_prompt: str = None
    
    def __init__(self, **data):
        # For backward compatibility, map default_prompt to prompt if provided
        if 'default_prompt' in data and not data.get('prompt'):
            data['prompt'] = data['default_prompt']
        super().__init__(**data)

class LLMService(BaseModel):
    """Configuration for an LLM service."""
    name: str
    type: str
    url: str
    models: List[LLMModel]

class LLMServiceManager:
    """Manager for LLM services."""
    
    def __init__(self, services: List[LLMService]):
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
        
    def get_service(self, name: str) -> Optional[LLMService]:
        """Get service configuration by name."""
        if not self._setup_called:
            self.setup()
        return self.services.get(name)
        
    def get_model(self, service_name: str, model_name: str) -> Optional[LLMModel]:
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

    def generate(
        self,
        service_name: str,
        model_name: str,
        input_text: str,
        timeout: float = 30.0,
        is_image: bool = False
    ) -> Optional[str]:
        """Generate text using specified service and model.
        
        Args:
            service_name: Name of the service to use
            model_name: Name of the model to use
            input_text: Text to process, or base64-encoded image if is_image=True
            timeout: Request timeout in seconds
            is_image: Whether input_text contains a base64-encoded image
            
        Returns:
            Generated text or None if failed
        """
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
            # Format prompt with input text
            prompt = model.prompt
            
            # Prepare request payload
            if is_image:
                self.logger.info(f"Processing image request for model {model.model}")
                
                # Clean up base64 data if needed
                if input_text.startswith('data:image'):
                    # Already a data URL - extract just the base64 part
                    base64_data = input_text.split(',')[1]
                    self.logger.debug("Extracted base64 from data URL format")
                else:
                    # Regular base64 string
                    base64_data = input_text
                    self.logger.debug("Using regular base64 format")
                
                # Prepare request with images array
                request_data = {
                    "model": model.model,
                    "prompt": prompt,  # Use prompt directly from model config
                    "images": [base64_data],
                    "stream": False
                }
                self.logger.info(f"Image request prepared for model {model.model}, base64 length: {len(base64_data)}")
            else:
                # Regular text input
                formatted_prompt = prompt.format(input_text=input_text)
                request_data = {
                    "model": model.model,
                    "prompt": formatted_prompt,
                    "stream": False
                }
                self.logger.debug("Using text format input")
            
            # Make request to Ollama API
            self.logger.debug(f"Sending request to {service.url}/api/generate")
            response = client.post(
                "/api/generate",
                json=request_data
            )
            response.raise_for_status()
            data = response.json()
            
            return data.get("response", "").strip()
            
        except Exception as e:
            self.logger.error(
                f"Failed to generate text: service={service_name}, "
                f"model={model_name}, error={str(e)}"
            )
            return None 