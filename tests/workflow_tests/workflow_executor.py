# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Workflow executor for running ComfyUI workflows via API.

Executes workflow JSON files through the ComfyUI server API and
retrieves results.
"""

import json
import urllib.request
import urllib.parse
import uuid
import time
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


class WorkflowExecutor:
    """Executes ComfyUI workflows via the server API."""

    def __init__(self, server_address="127.0.0.1:8188"):
        """
        Initialize workflow executor.

        Args:
            server_address: ComfyUI server address (host:port)
        """
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = None

    def connect(self):
        """Connect to ComfyUI websocket for real-time updates."""
        if not WEBSOCKET_AVAILABLE:
            raise ImportError("websocket-client package required for workflow execution")

        ws_url = f"ws://{self.server_address}/ws?clientId={self.client_id}"
        self.ws = websocket.WebSocket()
        self.ws.connect(ws_url)

    def disconnect(self):
        """Disconnect from ComfyUI websocket."""
        if self.ws:
            self.ws.close()
            self.ws = None

    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        """
        Queue a prompt for execution.

        Args:
            prompt: API format prompt

        Returns:
            Prompt ID for tracking
        """
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')

        req = urllib.request.Request(
            f"http://{self.server_address}/prompt",
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        response = urllib.request.urlopen(req)
        result = json.loads(response.read())

        if 'prompt_id' not in result:
            raise RuntimeError(f"Failed to queue prompt: {result}")

        return result['prompt_id']

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> bool:
        """
        Wait for prompt execution to complete.

        Args:
            prompt_id: Prompt ID to wait for
            timeout: Maximum wait time in seconds

        Returns:
            True if completed successfully
        """
        if not self.ws:
            raise RuntimeError("Not connected to websocket")

        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Workflow execution timed out after {timeout}s")

            out = self.ws.recv()

            if isinstance(out, str):
                message = json.loads(out)

                if message['type'] == 'executing':
                    data = message['data']

                    # When node is None and prompt_id matches, execution is done
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        return True

                elif message['type'] == 'execution_error':
                    error_data = message['data']
                    raise RuntimeError(f"Execution error: {error_data}")

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get execution history for a prompt.

        Args:
            prompt_id: Prompt ID

        Returns:
            History data including outputs
        """
        url = f"http://{self.server_address}/history/{prompt_id}"

        with urllib.request.urlopen(url) as response:
            history = json.loads(response.read())

        if prompt_id not in history:
            raise RuntimeError(f"Prompt {prompt_id} not found in history")

        return history[prompt_id]

    def get_outputs(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get outputs from executed workflow.

        Args:
            prompt_id: Prompt ID

        Returns:
            Dictionary of node outputs
        """
        history = self.get_history(prompt_id)
        return history.get('outputs', {})

    def execute_workflow(self, workflow: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """
        Execute a workflow and return outputs.

        Args:
            workflow: Workflow in API format
            timeout: Maximum execution time in seconds

        Returns:
            Dictionary of outputs by node ID
        """
        # Queue the workflow
        prompt_id = self.queue_prompt(workflow)

        # Wait for completion
        self.wait_for_completion(prompt_id, timeout)

        # Get and return outputs
        return self.get_outputs(prompt_id)

    def execute_workflow_file(self, workflow_path: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Load and execute a workflow JSON file.

        Args:
            workflow_path: Path to workflow JSON (UI or API format)
            timeout: Maximum execution time in seconds

        Returns:
            Dictionary of outputs by node ID
        """
        from .workflow_converter import WorkflowConverter

        # Load workflow
        with open(workflow_path, 'r') as f:
            workflow_json = json.load(f)

        # Check if it's UI format (has 'nodes' array) or API format
        if 'nodes' in workflow_json:
            # Convert from UI format to API format
            workflow = WorkflowConverter.convert_to_api_format(workflow_json)
        else:
            # Already in API format
            workflow = workflow_json

        return self.execute_workflow(workflow, timeout)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
