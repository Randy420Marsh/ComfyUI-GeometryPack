/**
 * ComfyUI GeomPack - Connected Components Dynamic Display
 * Shows component details in the node after execution
 */

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "geompack.connected_components",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackConnectedComponents") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create info panel container
                const infoPanel = document.createElement("div");
                infoPanel.style.backgroundColor = "#1a1a1a";
                infoPanel.style.padding = "8px";
                infoPanel.style.fontSize = "10px";
                infoPanel.style.fontFamily = "monospace";
                infoPanel.style.color = "#ccc";
                infoPanel.style.lineHeight = "1.4";
                infoPanel.style.maxHeight = "200px";
                infoPanel.style.overflowY = "auto";
                infoPanel.style.borderRadius = "4px";
                infoPanel.innerHTML = '<span style="color: #666;">Run workflow to see component details</span>';

                // Add widget
                const widget = this.addDOMWidget("component_info", "COMPONENT_INFO", infoPanel, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                widget.computeSize = () => [this.size[0] - 20, 120];

                this.componentInfoPanel = infoPanel;

                // Handle execution results
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    if (message?.component_data && message.component_data.length > 0) {
                        let html = '';

                        for (const meshData of message.component_data) {
                            const { mesh_name, num_components, total_faces, total_vertices, components } = meshData;

                            // Header with mesh name and totals
                            html += `<div style="margin-bottom: 8px;">`;
                            html += `<div style="color: #fff; font-weight: bold; margin-bottom: 4px;">`;
                            html += `${mesh_name}: <span style="color: #6cf;">${num_components}</span> component(s)`;
                            html += `</div>`;
                            html += `<div style="color: #888; font-size: 9px; margin-bottom: 4px;">`;
                            html += `Total: ${total_vertices.toLocaleString()} verts, ${total_faces.toLocaleString()} faces`;
                            html += `</div>`;

                            // Component table
                            html += `<table style="width: 100%; border-collapse: collapse; font-size: 9px;">`;
                            html += `<tr style="color: #888; border-bottom: 1px solid #333;">`;
                            html += `<th style="text-align: left; padding: 2px 4px;">#</th>`;
                            html += `<th style="text-align: right; padding: 2px 4px;">Faces</th>`;
                            html += `<th style="text-align: right; padding: 2px 4px;">Vertices</th>`;
                            html += `</tr>`;

                            // Show components (limit to 20 for UI performance)
                            const displayComponents = components.slice(0, 20);
                            const maxFaces = Math.max(...components.map(c => c.faces));
                            for (const comp of displayComponents) {
                                const color = comp.faces === maxFaces ? '#6f6' : '#ccc';

                                html += `<tr style="border-bottom: 1px solid #222;">`;
                                html += `<td style="padding: 2px 4px; color: #888;">${comp.id}</td>`;
                                html += `<td style="text-align: right; padding: 2px 4px; color: ${color};">${comp.faces.toLocaleString()}</td>`;
                                html += `<td style="text-align: right; padding: 2px 4px;">${comp.vertices.toLocaleString()}</td>`;
                                html += `</tr>`;

                                // Show face indices for small components (< 10 faces)
                                if (comp.face_indices && comp.face_indices.length > 0) {
                                    html += `<tr style="border-bottom: 1px solid #222;">`;
                                    html += `<td colspan="3" style="padding: 2px 4px 4px 16px; color: #888; font-size: 8px;">`;
                                    html += `faces: ${comp.face_indices.join(', ')}`;
                                    html += `</td></tr>`;
                                }
                            }

                            if (components.length > 20) {
                                html += `<tr><td colspan="3" style="padding: 4px; color: #888; text-align: center;">`;
                                html += `... and ${components.length - 20} more components`;
                                html += `</td></tr>`;
                            }

                            html += `</table>`;
                            html += `</div>`;
                        }

                        infoPanel.innerHTML = html;

                        // Resize widget based on content
                        const numRows = Math.min(message.component_data[0]?.components?.length || 0, 20) + 3;
                        const height = Math.min(Math.max(80, numRows * 16 + 40), 250);
                        widget.computeSize = () => [this.size[0] - 20, height];
                        this.setDirtyCanvas(true);
                    }
                };

                return r;
            };
        }
    }
});
