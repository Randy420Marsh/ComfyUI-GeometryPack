/**
 * ComfyUI-GeometryPack Transform Dynamic Parameter Management
 *
 * This extension manages the visibility of operation-specific parameters
 * in the Transform Mesh node based on the selected operation.
 */

import { app } from "../../../scripts/app.js";

const DEBUG = false; // Set to true to enable console logging

function log(...args) {
    if (DEBUG) {
        console.log("[GeomPack-Transform-Dynamic]", ...args);
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
    name: "geompack.transform_dynamic_parameters",

    async nodeCreated(node) {
        // Handle Transform Mesh node
        if (node.comfyClass === "GeomPackTransformMesh") {
            setTimeout(() => {
                this.setupTransformNode(node);
            }, 100);
        }
    },

    setupTransformNode(node) {
        log("Setting up GeomPackTransformMesh node");

        // Find the operation selector widget
        const operationWidget = node.widgets?.find(w => w.name === "operation");
        if (!operationWidget) {
            log("ERROR: Operation widget not found!");
            return;
        }

        log("Operation widget found, current value:", operationWidget.value);

        // Find all parameter widgets
        const translateXWidget = node.widgets?.find(w => w.name === "translate_x");
        const translateYWidget = node.widgets?.find(w => w.name === "translate_y");
        const translateZWidget = node.widgets?.find(w => w.name === "translate_z");

        const rotateXWidget = node.widgets?.find(w => w.name === "rotate_x");
        const rotateYWidget = node.widgets?.find(w => w.name === "rotate_y");
        const rotateZWidget = node.widgets?.find(w => w.name === "rotate_z");

        const scaleUniformWidget = node.widgets?.find(w => w.name === "scale_uniform");
        const scaleXWidget = node.widgets?.find(w => w.name === "scale_x");
        const scaleYWidget = node.widgets?.find(w => w.name === "scale_y");
        const scaleZWidget = node.widgets?.find(w => w.name === "scale_z");

        const mirrorAxisWidget = node.widgets?.find(w => w.name === "mirror_axis");

        const centerXWidget = node.widgets?.find(w => w.name === "center_x");
        const centerYWidget = node.widgets?.find(w => w.name === "center_y");
        const centerZWidget = node.widgets?.find(w => w.name === "center_z");

        const matrixStringWidget = node.widgets?.find(w => w.name === "matrix_string");

        log("Found parameter widgets:", {
            translate_x: !!translateXWidget,
            translate_y: !!translateYWidget,
            translate_z: !!translateZWidget,
            rotate_x: !!rotateXWidget,
            rotate_y: !!rotateYWidget,
            rotate_z: !!rotateZWidget,
            scale_uniform: !!scaleUniformWidget,
            scale_x: !!scaleXWidget,
            scale_y: !!scaleYWidget,
            scale_z: !!scaleZWidget,
            mirror_axis: !!mirrorAxisWidget,
            center_x: !!centerXWidget,
            center_y: !!centerYWidget,
            center_z: !!centerZWidget,
            matrix_string: !!matrixStringWidget
        });

        // Function to update widget visibility based on selected operation
        const updateWidgetVisibility = (selectedOperation) => {
            log("Updating widget visibility for operation:", selectedOperation);

            // Hide all optional parameters first
            hideWidget(node, translateXWidget);
            hideWidget(node, translateYWidget);
            hideWidget(node, translateZWidget);
            hideWidget(node, rotateXWidget);
            hideWidget(node, rotateYWidget);
            hideWidget(node, rotateZWidget);
            hideWidget(node, scaleUniformWidget);
            hideWidget(node, scaleXWidget);
            hideWidget(node, scaleYWidget);
            hideWidget(node, scaleZWidget);
            hideWidget(node, mirrorAxisWidget);
            hideWidget(node, centerXWidget);
            hideWidget(node, centerYWidget);
            hideWidget(node, centerZWidget);
            hideWidget(node, matrixStringWidget);

            // Show relevant parameters based on operation
            if (selectedOperation === "translate") {
                showWidget(node, translateXWidget);
                showWidget(node, translateYWidget);
                showWidget(node, translateZWidget);
            } else if (selectedOperation === "rotate") {
                showWidget(node, rotateXWidget);
                showWidget(node, rotateYWidget);
                showWidget(node, rotateZWidget);
            } else if (selectedOperation === "scale") {
                showWidget(node, scaleUniformWidget);
                showWidget(node, scaleXWidget);
                showWidget(node, scaleYWidget);
                showWidget(node, scaleZWidget);
            } else if (selectedOperation === "mirror") {
                showWidget(node, mirrorAxisWidget);
            } else if (selectedOperation === "center") {
                showWidget(node, centerXWidget);
                showWidget(node, centerYWidget);
                showWidget(node, centerZWidget);
            } else if (selectedOperation === "align_to_axes") {
                // No parameters needed - all hidden
            } else if (selectedOperation === "apply_matrix") {
                showWidget(node, matrixStringWidget);
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

log("GeomPack transform dynamic parameters extension loaded");
