# Workflow-Based Testing

This directory contains infrastructure for executing and testing ComfyUI workflows via the ComfyUI API.

## Overview

Unlike traditional unit tests that instantiate nodes directly, these tests execute actual workflow JSON files through a running ComfyUI server. This provides end-to-end validation of the complete workflow execution path.

## Architecture

### Core Components

**WorkflowExecutor** (`workflow_executor.py`)
- Connects to ComfyUI server via websocket
- Queues workflow prompts for execution
- Waits for completion and retrieves outputs
- Usage:
  ```python
  with WorkflowExecutor() as executor:
      outputs = executor.execute_workflow_file("workflow.json")
  ```

**WorkflowConverter** (`workflow_converter.py`)
- Converts UI workflow JSON format to API format
- Maps widget values to input parameters
- Resolves link connections between nodes
- Usage:
  ```python
  from workflow_converter import WorkflowConverter
  api_prompt = WorkflowConverter.convert_to_api_format(workflow_json)
  ```

**MeshValidator** (`mesh_validators.py`)
- Validates mesh geometry and topology
- Checks vertex/face counts, bounding boxes, UV coordinates
- Provides assertions for test validation
- Usage:
  ```python
  is_valid, error = MeshValidator.is_valid_mesh(mesh)
  assert is_valid, error
  ```

## Running Tests

### Prerequisites

1. **Start ComfyUI server:**
   ```bash
   cd /path/to/ComfyUI
   python main.py --listen 127.0.0.1 --port 8188
   ```

2. **Install test dependencies:**
   ```bash
   pip install pytest websocket-client
   ```

### Run Workflow Tests

```bash
# Run all workflow tests
pytest tests/workflow_tests/ -v -m workflow

# Run only simple workflow tests
pytest tests/workflow_tests/ -v -m workflow_simple

# Run a specific workflow test
pytest tests/workflow_tests/test_simple_workflows.py::test_preview_mesh_workflow -v

# Run unit tests (no ComfyUI server needed)
pytest tests/workflow_tests/ -v -m unit
```

## Test Markers

Tests use pytest markers for categorization:

- `@pytest.mark.workflow` - All workflow execution tests
- `@pytest.mark.workflow_simple` - Simple workflows (2-3 nodes)
- `@pytest.mark.workflow_medium` - Medium workflows (4-5 nodes)
- `@pytest.mark.workflow_complex` - Complex workflows (6-7 nodes)
- `@pytest.mark.unit` - Unit tests (no server required)

## Adding New Workflow Tests

1. **Add workflow JSON to `/workflows/` directory**

2. **Add to test parameters:**
   ```python
   SIMPLE_WORKFLOWS = [
       "preview_mesh.json",
       "your_new_workflow.json",  # Add here
   ]
   ```

3. **Optionally add specific test:**
   ```python
   @pytest.mark.workflow
   def test_your_workflow(workflow_executor):
       outputs = workflow_executor.execute_workflow_file("your_workflow.json")
       assert outputs is not None
       # Add custom validations
   ```

## Workflow Format

Workflows can be in two formats:

**UI Format** (from ComfyUI save):
```json
{
  "nodes": [
    {
      "id": 1,
      "type": "GeomPackLoadMesh",
      "widgets_values": ["bunny.stl"],
      "inputs": [],
      "outputs": [{"name": "mesh", "links": [1]}]
    }
  ],
  "links": [[1, 1, 0, 2, 0, "TRIMESH"]]
}
```

**API Format** (converted automatically):
```json
{
  "1": {
    "class_type": "GeomPackLoadMesh",
    "inputs": {
      "filename": "bunny.stl"
    }
  }
}
```

## Troubleshooting

**"ComfyUI server not available"**
- Ensure ComfyUI is running at 127.0.0.1:8188
- Check firewall settings
- Verify ComfyUI started without errors

**"Workflow execution failed"**
- Check ComfyUI server logs for errors
- Verify all required nodes are installed
- Check that input files exist in ComfyUI input directory

**"websocket-client package required"**
- Install: `pip install websocket-client`

## CI/CD Integration

For GitHub Actions, see `.github/workflows/test-workflows.yml`:
- Installs ComfyUI
- Starts server in background
- Runs workflow tests
- Uploads output artifacts

## Future Enhancements

- Golden reference comparison (compare outputs to baseline meshes)
- Visual diff generation (render and compare images)
- Performance benchmarking
- Parallel workflow execution
- Test result dashboard
