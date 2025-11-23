# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Workflow JSON to API format converter.

Converts ComfyUI UI workflow JSON format to the API prompt format
required by the ComfyUI server.
"""

import json
from typing import Dict, Any, List


class WorkflowConverter:
    """Converts workflow UI JSON to API format."""

    @staticmethod
    def convert_to_api_format(workflow_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert UI workflow JSON to API prompt format.

        Args:
            workflow_json: Workflow in UI format (has 'nodes' and 'links' arrays)

        Returns:
            Dict in API format (node IDs as keys)
        """
        api_prompt = {}

        # Build link lookup table: link_id -> (source_node, source_output_slot)
        link_map = {}
        for link in workflow_json.get('links', []):
            # Link format: [link_id, source_node, source_slot, dest_node, dest_slot, type]
            link_id = link[0]
            source_node = link[1]
            source_slot = link[2]
            link_map[link_id] = (source_node, source_slot)

        # Convert each node
        for node in workflow_json.get('nodes', []):
            node_id = str(node['id'])
            node_type = node['type']

            api_node = {
                'class_type': node_type,
                'inputs': {}
            }

            # Process input connections (from links)
            for input_def in node.get('inputs', []):
                input_name = input_def['name']
                link_id = input_def.get('link')

                if link_id is not None and link_id in link_map:
                    source_node, source_slot = link_map[link_id]
                    api_node['inputs'][input_name] = [str(source_node), source_slot]

            # Process widget values (direct inputs)
            widget_values = node.get('widgets_values', [])
            if widget_values:
                # Map widget values to input names
                # This requires knowing the node's INPUT_TYPES, but we'll use a heuristic:
                # Get input names that don't have links (they're widget inputs)
                linked_inputs = set(inp['name'] for inp in node.get('inputs', []) if inp.get('link') is not None)

                # Try to map widget values to non-linked inputs by order
                widget_input_names = [inp['name'] for inp in node.get('inputs', []) if inp['name'] not in linked_inputs]

                for i, widget_value in enumerate(widget_values):
                    if i < len(widget_input_names):
                        api_node['inputs'][widget_input_names[i]] = widget_value
                    else:
                        # If we have more widget values than expected inputs,
                        # use generic names (this might not work perfectly)
                        api_node['inputs'][f'widget_{i}'] = widget_value

            api_prompt[node_id] = api_node

        return api_prompt

    @staticmethod
    def load_workflow(workflow_path: str) -> Dict[str, Any]:
        """
        Load workflow JSON from file.

        Args:
            workflow_path: Path to workflow JSON file

        Returns:
            Workflow dictionary
        """
        with open(workflow_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_api_prompt(api_prompt: Dict[str, Any], output_path: str):
        """
        Save API prompt to JSON file.

        Args:
            api_prompt: API format prompt
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(api_prompt, f, indent=2)

    @classmethod
    def convert_workflow_file(cls, workflow_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Load workflow from file, convert to API format, and optionally save.

        Args:
            workflow_path: Path to workflow JSON file
            output_path: Optional path to save API format

        Returns:
            API format prompt
        """
        workflow = cls.load_workflow(workflow_path)
        api_prompt = cls.convert_to_api_format(workflow)

        if output_path:
            cls.save_api_prompt(api_prompt, output_path)

        return api_prompt
