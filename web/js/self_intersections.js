/**
 * ComfyUI GeomPack - Self Intersections Dynamic Display
 * Shows self-intersecting faces in the node after execution
 */

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "geompack.self_intersections",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackDetectSelfIntersections") {

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
                infoPanel.innerHTML = '<span style="color: #666;">Run workflow to detect self-intersections</span>';

                // Add widget
                const widget = this.addDOMWidget("intersection_info", "INTERSECTION_INFO", infoPanel, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                widget.computeSize = () => [this.size[0] - 20, 120];

                this.intersectionInfoPanel = infoPanel;

                // Handle execution results
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    if (message?.intersection_data && message.intersection_data.length > 0) {
                        let html = '';

                        for (const data of message.intersection_data) {
                            const { mesh_name, num_intersecting_faces, num_intersection_pairs, total_faces, total_vertices, has_cgal, faces } = data;

                            // Header with mesh name and status
                            html += `<div style="margin-bottom: 8px;">`;

                            // Color based on whether there are intersections
                            const statusColor = num_intersecting_faces === 0 ? '#6f6' : '#f66';
                            const statusText = num_intersecting_faces === 0 ? 'Clean' : `${num_intersecting_faces} intersecting face(s)`;

                            html += `<div style="color: #fff; font-weight: bold; margin-bottom: 4px;">`;
                            html += `${mesh_name}: <span style="color: ${statusColor};">${statusText}</span>`;
                            html += `</div>`;

                            if (!has_cgal) {
                                html += `<div style="color: #f96; font-size: 9px; margin-bottom: 4px;">`;
                                html += `CGAL not available - results may be incomplete`;
                                html += `</div>`;
                            }

                            if (num_intersecting_faces > 0) {
                                html += `<div style="color: #888; font-size: 9px; margin-bottom: 4px;">`;
                                html += `${num_intersection_pairs} intersection pair(s) detected`;
                                html += `</div>`;

                                // Face table
                                html += `<table style="width: 100%; border-collapse: collapse; font-size: 9px;">`;
                                html += `<tr style="color: #888; border-bottom: 1px solid #333;">`;
                                html += `<th style="text-align: left; padding: 2px 4px;">Face</th>`;
                                html += `<th style="text-align: left; padding: 2px 4px;">Vertices</th>`;
                                html += `</tr>`;

                                // Show faces (limit to 20 for UI performance)
                                const displayFaces = faces.slice(0, 20);
                                for (const face of displayFaces) {
                                    html += `<tr style="border-bottom: 1px solid #222;">`;
                                    html += `<td style="padding: 2px 4px; color: #f66;">${face.id}</td>`;
                                    html += `<td style="padding: 2px 4px; color: #888;">[${face.vertices.join(', ')}]</td>`;
                                    html += `</tr>`;
                                }

                                if (faces.length > 20) {
                                    html += `<tr><td colspan="2" style="padding: 4px; color: #888; text-align: center;">`;
                                    html += `... and ${faces.length - 20} more faces`;
                                    html += `</td></tr>`;
                                }

                                html += `</table>`;
                            } else {
                                html += `<div style="color: #888; font-size: 9px;">`;
                                html += `Total: ${total_vertices.toLocaleString()} vertices, ${total_faces.toLocaleString()} faces`;
                                html += `</div>`;
                            }

                            html += `</div>`;
                        }

                        infoPanel.innerHTML = html;

                        // Resize widget based on content
                        const numRows = Math.min(message.intersection_data[0]?.faces?.length || 0, 20) + 3;
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
