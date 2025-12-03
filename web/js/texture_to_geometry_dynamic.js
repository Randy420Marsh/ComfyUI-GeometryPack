/**
 * ComfyUI-GeometryPack Dynamic Parameter Management
 *
 * This extension manages the visibility of backend-specific parameters
 * in the Depth Map to Mesh node based on the selected backend.
 */

import { app } from "../../../scripts/app.js";

const DEBUG = false;

function log(...args) {
    if (DEBUG) {
        console.log("[GeomPack-TextureToGeometry]", ...args);
    }
}

// Helper function to hide a widget
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

    if (!widget.origType) {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
        widget.origSerializeValue = widget.serializeValue;
    }

    widget._originalIndex = index;
    widget._hidden = true;

    node.widgets.splice(index, 1);

    log("  Widget removed from index:", index);

    if (widget.linkedWidgets) {
        widget.linkedWidgets.forEach(w => hideWidget(node, w));
    }
}

// Helper function to show a widget
function showWidget(node, widget) {
    if (!widget) return;
    if (!widget._hidden) {
        log("Widget already visible:", widget.name);
        return;
    }

    log("Showing widget:", widget.name);

    if (widget.origType) {
        widget.type = widget.origType;
        widget.computeSize = widget.origComputeSize;
        if (widget.origSerializeValue) {
            widget.serializeValue = widget.origSerializeValue;
        }
    }

    const insertIndex = Math.min(widget._originalIndex, node.widgets.length);
    node.widgets.splice(insertIndex, 0, widget);

    log("  Widget restored to index:", insertIndex);

    widget._hidden = false;

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
    name: "geompack.texture_to_geometry_dynamic",

    async nodeCreated(node) {
        if (node.comfyClass === "GeomPackTextureToGeometry") {
            setTimeout(() => {
                this.setupTextureToGeometryNode(node);
            }, 100);
        }
    },

    setupTextureToGeometryNode(node) {
        log("Setting up GeomPackTextureToGeometry node");

        // Find the backend selector widget
        const backendWidget = node.widgets?.find(w => w.name === "backend");
        if (!backendWidget) {
            log("ERROR: Backend widget not found!");
            return;
        }

        log("Backend widget found, current value:", backendWidget.value);

        // Find all parameter widgets
        const poissonDepthWidget = node.widgets?.find(w => w.name === "poisson_depth");
        const smoothNormalsWidget = node.widgets?.find(w => w.name === "smooth_normals");

        log("Found parameter widgets:", {
            poisson_depth: !!poissonDepthWidget,
            smooth_normals: !!smoothNormalsWidget
        });

        // Function to update widget visibility based on selected backend
        const updateWidgetVisibility = (selectedBackend) => {
            log("Updating widget visibility for backend:", selectedBackend);

            // Hide all backend-specific parameters first
            hideWidget(node, poissonDepthWidget);
            hideWidget(node, smoothNormalsWidget);

            // Show relevant parameters based on backend
            if (selectedBackend === "grid") {
                // Grid: show smooth_normals, hide poisson_depth
                showWidget(node, smoothNormalsWidget);
            } else if (selectedBackend === "poisson_pymeshlab" || selectedBackend === "poisson_open3d") {
                // Poisson backends: show poisson_depth, hide smooth_normals
                showWidget(node, poissonDepthWidget);
            } else if (selectedBackend === "delaunay_2d") {
                // Delaunay: hide both poisson_depth and smooth_normals
                // (already hidden above)
            }

            updateNodeSize(node);
        };

        // Store original callback
        const origCallback = backendWidget.callback;

        // Override callback to update visibility when backend changes
        backendWidget.callback = function(value) {
            log("Backend changed to:", value);

            const result = origCallback?.apply(this, arguments);

            updateWidgetVisibility(value);

            return result;
        };

        // Initialize visibility on node creation
        log("Initializing widget visibility...");
        updateWidgetVisibility(backendWidget.value);
    }
});

log("GeomPack TextureToGeometry dynamic parameters extension loaded");
