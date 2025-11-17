/**
 * ComfyUI-GeometryPack Dynamic Parameter Management
 *
 * This extension manages the visibility of backend-specific parameters
 * in the Remesh and Refine Mesh nodes based on the selected backend/operation.
 */

import { app } from "../../../scripts/app.js";

const DEBUG = false; // Set to true to enable console logging

function log(...args) {
    if (DEBUG) {
        console.log("[GeomPack-Dynamic]", ...args);
    }
}

// Helper function to hide a widget (removes from array)
function hideWidget(node, widget) {
    if (!widget) return;
    if (widget._hidden) {
        log("Widget already hidden:", widget.name);
        return;
    }

    log("Hiding widget:", widget.name);

    const index = node.widgets.indexOf(widget);
    if (index === -1) {
        log("  ERROR: Widget not found in array!");
        return;
    }

    // Store original properties for restoration
    if (!widget.origType) {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
        widget.origSerializeValue = widget.serializeValue;
    }

    widget._originalIndex = index;
    widget._hidden = true;

    // Remove widget from array
    node.widgets.splice(index, 1);

    log("  Widget removed from index:", index);

    // Hide linked widgets if any
    if (widget.linkedWidgets) {
        widget.linkedWidgets.forEach(w => hideWidget(node, w));
    }
}

// Helper function to show a widget (re-inserts into array)
function showWidget(node, widget) {
    if (!widget) return;
    if (!widget._hidden) {
        log("Widget already visible:", widget.name);
        return;
    }

    log("Showing widget:", widget.name);

    // Restore original properties
    if (widget.origType) {
        widget.type = widget.origType;
        widget.computeSize = widget.origComputeSize;
        if (widget.origSerializeValue) {
            widget.serializeValue = widget.origSerializeValue;
        }
    }

    // Re-insert widget at original position
    const insertIndex = Math.min(widget._originalIndex, node.widgets.length);
    node.widgets.splice(insertIndex, 0, widget);

    log("  Widget restored to index:", insertIndex);

    widget._hidden = false;

    // Show linked widgets if any
    if (widget.linkedWidgets) {
        widget.linkedWidgets.forEach(w => showWidget(node, w));
    }
}

// Helper to update node size and redraw
function updateNodeSize(node) {
    node.setDirtyCanvas(true, true);
    if (app.graph) {
        app.graph.setDirtyCanvas(true, true);
    }

    requestAnimationFrame(() => {
        const newSize = node.computeSize();
        node.setSize([node.size[0], newSize[1]]);
        node.setDirtyCanvas(true, true);

        if (app.canvas) {
            app.canvas.setDirty(true, true);
        }

        requestAnimationFrame(() => {
            if (app.canvas) {
                app.canvas.draw(true, true);
            }
            log("Widget visibility update complete");
        });
    });
}

// Main extension registration
app.registerExtension({
    name: "geompack.dynamic_parameters",

    async nodeCreated(node) {
        // Handle Remesh node
        if (node.comfyClass === "GeomPackRemesh") {
            setTimeout(() => {
                this.setupRemeshNode(node);
            }, 100);
        }

        // Handle Refine Mesh node
        if (node.comfyClass === "GeomPackRefineMesh") {
            setTimeout(() => {
                this.setupRefineMeshNode(node);
            }, 100);
        }
    },

    setupRemeshNode(node) {
        log("Setting up GeomPackRemesh node");

        // Find the backend selector widget
        const backendWidget = node.widgets?.find(w => w.name === "backend");
        if (!backendWidget) {
            log("ERROR: Backend widget not found!");
            return;
        }

        log("Backend widget found, current value:", backendWidget.value);

        // Find all parameter widgets
        const targetEdgeLengthWidget = node.widgets?.find(w => w.name === "target_edge_length");
        const iterationsWidget = node.widgets?.find(w => w.name === "iterations");
        const protectBoundariesWidget = node.widgets?.find(w => w.name === "protect_boundaries");
        const voxelSizeWidget = node.widgets?.find(w => w.name === "voxel_size");
        const targetFaceCountWidget = node.widgets?.find(w => w.name === "target_face_count");
        const targetVertexCountWidget = node.widgets?.find(w => w.name === "target_vertex_count");
        const deterministicWidget = node.widgets?.find(w => w.name === "deterministic");
        const creaseAngleWidget = node.widgets?.find(w => w.name === "crease_angle");

        log("Found parameter widgets:", {
            target_edge_length: !!targetEdgeLengthWidget,
            iterations: !!iterationsWidget,
            protect_boundaries: !!protectBoundariesWidget,
            voxel_size: !!voxelSizeWidget,
            target_face_count: !!targetFaceCountWidget,
            target_vertex_count: !!targetVertexCountWidget,
            deterministic: !!deterministicWidget,
            crease_angle: !!creaseAngleWidget
        });

        // Function to update widget visibility based on selected backend
        const updateWidgetVisibility = (selectedBackend) => {
            log("Updating widget visibility for backend:", selectedBackend);

            // Hide all optional parameters first
            hideWidget(node, targetEdgeLengthWidget);
            hideWidget(node, iterationsWidget);
            hideWidget(node, protectBoundariesWidget);
            hideWidget(node, voxelSizeWidget);
            hideWidget(node, targetFaceCountWidget);
            hideWidget(node, targetVertexCountWidget);
            hideWidget(node, deterministicWidget);
            hideWidget(node, creaseAngleWidget);

            // Show relevant parameters based on backend
            if (selectedBackend === "pymeshlab_isotropic") {
                showWidget(node, targetEdgeLengthWidget);
                showWidget(node, iterationsWidget);
            } else if (selectedBackend === "cgal_isotropic") {
                showWidget(node, targetEdgeLengthWidget);
                showWidget(node, iterationsWidget);
                showWidget(node, protectBoundariesWidget);
            } else if (selectedBackend === "blender_voxel") {
                showWidget(node, voxelSizeWidget);
            } else if (selectedBackend === "blender_quadriflow") {
                showWidget(node, targetFaceCountWidget);
            } else if (selectedBackend === "instant_meshes") {
                showWidget(node, targetVertexCountWidget);
                showWidget(node, deterministicWidget);
                showWidget(node, creaseAngleWidget);
            }

            updateNodeSize(node);
        };

        // Store original callback
        const origCallback = backendWidget.callback;

        // Override callback to update visibility when backend changes
        backendWidget.callback = function(value) {
            log("Backend changed to:", value);

            // Call original callback if it exists
            const result = origCallback?.apply(this, arguments);

            // Update widget visibility
            updateWidgetVisibility(value);

            return result;
        };

        // Initialize visibility on node creation
        log("Initializing widget visibility...");
        updateWidgetVisibility(backendWidget.value);
    },

    setupRefineMeshNode(node) {
        log("Setting up GeomPackRefineMesh node");

        // Find the operation selector widget
        const operationWidget = node.widgets?.find(w => w.name === "operation");
        if (!operationWidget) {
            log("ERROR: Operation widget not found!");
            return;
        }

        log("Operation widget found, current value:", operationWidget.value);

        // Find all parameter widgets
        const targetFaceCountWidget = node.widgets?.find(w => w.name === "target_face_count");
        const decimationMethodWidget = node.widgets?.find(w => w.name === "decimation_method");
        const subdivisionIterationsWidget = node.widgets?.find(w => w.name === "subdivision_iterations");
        const smoothingIterationsWidget = node.widgets?.find(w => w.name === "smoothing_iterations");
        const lambdaFactorWidget = node.widgets?.find(w => w.name === "lambda_factor");

        log("Found parameter widgets:", {
            target_face_count: !!targetFaceCountWidget,
            decimation_method: !!decimationMethodWidget,
            subdivision_iterations: !!subdivisionIterationsWidget,
            smoothing_iterations: !!smoothingIterationsWidget,
            lambda_factor: !!lambdaFactorWidget
        });

        // Function to update widget visibility based on selected operation
        const updateWidgetVisibility = (selectedOperation) => {
            log("Updating widget visibility for operation:", selectedOperation);

            // Hide all optional parameters first
            hideWidget(node, targetFaceCountWidget);
            hideWidget(node, decimationMethodWidget);
            hideWidget(node, subdivisionIterationsWidget);
            hideWidget(node, smoothingIterationsWidget);
            hideWidget(node, lambdaFactorWidget);

            // Show relevant parameters based on operation
            if (selectedOperation === "decimation") {
                showWidget(node, targetFaceCountWidget);
                showWidget(node, decimationMethodWidget);
            } else if (selectedOperation === "subdivision_loop" || selectedOperation === "subdivision_midpoint") {
                showWidget(node, subdivisionIterationsWidget);
            } else if (selectedOperation === "laplacian_smoothing") {
                showWidget(node, smoothingIterationsWidget);
                showWidget(node, lambdaFactorWidget);
            }

            updateNodeSize(node);
        };

        // Store original callback
        const origCallback = operationWidget.callback;

        // Override callback to update visibility when operation changes
        operationWidget.callback = function(value) {
            log("Operation changed to:", value);

            // Call original callback if it exists
            const result = origCallback?.apply(this, arguments);

            // Update widget visibility
            updateWidgetVisibility(value);

            return result;
        };

        // Initialize visibility on node creation
        log("Initializing widget visibility...");
        updateWidgetVisibility(operationWidget.value);
    }
});

log("GeomPack dynamic parameters extension loaded");
