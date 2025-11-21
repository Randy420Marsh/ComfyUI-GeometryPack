/**
 * ComfyUI GeomPack - Dual Mesh Preview Widget
 * Unified viewer for side-by-side and overlay dual mesh visualization
 * with full field visualization support
 */

import { app } from "../../../scripts/app.js";

console.log('[GeomPack Dual JS] Loading mesh_preview_dual.js extension - v2 WITH INCREASED NODE HEIGHT (680px)');

app.registerExtension({
    name: "geompack.meshpreview.dual",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log('[GeomPack Dual JS] beforeRegisterNodeDef called for:', nodeData.name);
        if (nodeData.name === "GeomPackPreviewMeshDual") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log('[GeomPack Dual JS] Creating PreviewMeshDual node widget');

                // Create container for viewer + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#2a2a2a";

                // Create iframe for VTK.js viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1";
                iframe.style.minHeight = "550px";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";

                console.log('[GeomPack Dual] Created iframe with minHeight:', iframe.style.minHeight);

                // Point to unified dual VTK.js HTML viewer (with cache buster)
                iframe.src = "/extensions/ComfyUI-GeometryPack/viewer_dual.html?v=" + Date.now();

                // Create mesh info panel
                const infoPanel = document.createElement("div");
                infoPanel.style.backgroundColor = "#1a1a1a";
                infoPanel.style.borderTop = "1px solid #444";
                infoPanel.style.padding = "6px 12px";
                infoPanel.style.fontSize = "10px";
                infoPanel.style.fontFamily = "monospace";
                infoPanel.style.color = "#ccc";
                infoPanel.style.lineHeight = "1.3";
                infoPanel.style.flexShrink = "0";
                infoPanel.style.overflow = "hidden";
                infoPanel.innerHTML = '<span style="color: #888;">Mesh info will appear here after execution</span>';

                // Add iframe and info panel to container
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget with required options
                const widget = this.addDOMWidget("preview_dual", "MESH_PREVIEW_DUAL", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                // Default widget size (will be wider for side-by-side)
                widget.computeSize = () => [768, 680];

                // Store iframe and info panel references
                this.meshViewerIframeDual = iframe;
                this.meshInfoPanelDual = infoPanel;

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    iframeLoaded = true;
                    console.log('[GeomPack Dual] Iframe loaded successfully');
                    console.log('[GeomPack Dual] Iframe dimensions:', {
                        width: iframe.offsetWidth,
                        height: iframe.offsetHeight,
                        clientWidth: iframe.clientWidth,
                        clientHeight: iframe.clientHeight
                    });
                });

                // Listen for messages from iframe
                window.addEventListener('message', async (event) => {
                    // Handle error messages from iframe
                    if (event.data.type === 'MESH_ERROR' && event.data.error) {
                        console.error('[GeomPack Dual] Error from viewer:', event.data.error);
                        if (infoPanel) {
                            infoPanel.innerHTML = `<div style="color: #ff6b6b; padding: 8px;">Error loading mesh: ${event.data.error}</div>`;
                        }
                    }
                });

                // Set initial node size
                this.setSize([768, 680]);

                // Debug: Log container and iframe info after setup
                setTimeout(() => {
                    console.log('[GeomPack Dual JS] Container dimensions:', {
                        width: container.offsetWidth,
                        height: container.offsetHeight,
                        clientWidth: container.clientWidth,
                        clientHeight: container.clientHeight
                    });
                    console.log('[GeomPack Dual JS] Iframe dimensions after setup:', {
                        width: iframe.offsetWidth,
                        height: iframe.offsetHeight,
                        clientWidth: iframe.clientWidth,
                        clientHeight: iframe.clientHeight,
                        minHeight: iframe.style.minHeight,
                        flex: iframe.style.flex
                    });
                }, 500);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    if (!message?.layout) {
                        return;
                    }

                    const layout = message.layout[0];

                    let infoHTML = '';
                    let postMessageData = {
                        type: 'LOAD_DUAL_MESH',
                        layout: layout,
                        timestamp: Date.now()
                    };

                    if (layout === 'side_by_side') {
                        // Side-by-side mode
                        if (!message?.mesh_1_file || !message?.mesh_2_file) {
                            return;
                        }

                        const filename1 = message.mesh_1_file[0];
                        const filename2 = message.mesh_2_file[0];

                        // Update mesh info panel
                        const vertices1 = message.vertex_count_1?.[0] || 'N/A';
                        const vertices2 = message.vertex_count_2?.[0] || 'N/A';
                        const faces1 = message.face_count_1?.[0] || 'N/A';
                        const faces2 = message.face_count_2?.[0] || 'N/A';

                        const extents1 = message.extents_1?.[0] || [];
                        const extents2 = message.extents_2?.[0] || [];

                        const extentsStr1 = extents1.length === 3 ?
                            `${extents1.map(v => v.toFixed(2)).join(' × ')}` : 'N/A';
                        const extentsStr2 = extents2.length === 3 ?
                            `${extents2.map(v => v.toFixed(2)).join(' × ')}` : 'N/A';

                        infoHTML = `
                            <div style="display: grid; grid-template-columns: auto 1fr 1fr; gap: 2px 12px;">
                                <span style="color: #888;"></span>
                                <span style="color: #999; font-weight: bold; border-bottom: 1px solid #333;">Mesh 1</span>
                                <span style="color: #999; font-weight: bold; border-bottom: 1px solid #333;">Mesh 2</span>

                                <span style="color: #888;">Vertices:</span>
                                <span>${vertices1.toLocaleString()}</span>
                                <span>${vertices2.toLocaleString()}</span>

                                <span style="color: #888;">Faces:</span>
                                <span>${faces1.toLocaleString()}</span>
                                <span>${faces2.toLocaleString()}</span>

                                <span style="color: #888;">Extents:</span>
                                <span style="font-size: 9px;">${extentsStr1}</span>
                                <span style="font-size: 9px;">${extentsStr2}</span>
                        `;

                        // Add watertight info if available
                        if (message.is_watertight_1 !== undefined && message.is_watertight_2 !== undefined) {
                            const watertight1 = message.is_watertight_1[0] ? 'Yes' : 'No';
                            const watertight2 = message.is_watertight_2[0] ? 'Yes' : 'No';
                            const color1 = message.is_watertight_1[0] ? '#6c6' : '#c66';
                            const color2 = message.is_watertight_2[0] ? '#6c6' : '#c66';
                            infoHTML += `
                                <span style="color: #888;">Watertight:</span>
                                <span style="color: ${color1};">${watertight1}</span>
                                <span style="color: ${color2};">${watertight2}</span>
                            `;
                        }

                        // Add field info if available
                        if (message.common_fields && message.common_fields[0].length > 0) {
                            const commonFields = message.common_fields[0];
                            infoHTML += `
                                <span style="color: #888;">Fields:</span>
                                <span colspan="2" style="grid-column: 2 / 4; color: #9c9;">${commonFields.length} shared field(s)</span>
                            `;
                        }

                        infoHTML += '</div>';

                        // Prepare file paths
                        const filepath1 = `/view?filename=${encodeURIComponent(filename1)}&type=output&subfolder=`;
                        const filepath2 = `/view?filename=${encodeURIComponent(filename2)}&type=output&subfolder=`;

                        postMessageData.mesh1Filepath = filepath1;
                        postMessageData.mesh2Filepath = filepath2;

                    } else {
                        // Overlay mode
                        if (!message?.mesh_file) {
                            return;
                        }

                        const filename = message.mesh_file[0];

                        // Update mesh info panel
                        const vertices1 = message.vertex_count_1?.[0] || 'N/A';
                        const vertices2 = message.vertex_count_2?.[0] || 'N/A';
                        const faces1 = message.face_count_1?.[0] || 'N/A';
                        const faces2 = message.face_count_2?.[0] || 'N/A';

                        infoHTML = `
                            <div style="display: grid; grid-template-columns: auto 1fr 1fr; gap: 2px 12px;">
                                <span style="color: #888;"></span>
                                <span style="color: #999; font-weight: bold; border-bottom: 1px solid #333;">Mesh 1</span>
                                <span style="color: #999; font-weight: bold; border-bottom: 1px solid #333;">Mesh 2</span>

                                <span style="color: #888;">Vertices:</span>
                                <span>${vertices1.toLocaleString()}</span>
                                <span>${vertices2.toLocaleString()}</span>

                                <span style="color: #888;">Faces:</span>
                                <span>${faces1.toLocaleString()}</span>
                                <span>${faces2.toLocaleString()}</span>
                        `;

                        // Add color info if using vertex coloring
                        if (message.mesh_1_color && message.mesh_2_color) {
                            infoHTML += `
                                <span style="color: #888;">Colors:</span>
                                <span>${message.mesh_1_color[0]}</span>
                                <span>${message.mesh_2_color[0]}</span>
                            `;
                        }

                        infoHTML += '</div>';

                        // Prepare file path
                        const filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

                        postMessageData.meshFilepath = filepath;
                    }

                    infoPanel.innerHTML = infoHTML;

                    // Function to send message
                    const sendMessage = () => {
                        if (iframe.contentWindow) {
                            console.log('[GeomPack Dual] Sending message to iframe:', postMessageData);
                            iframe.contentWindow.postMessage(postMessageData, "*");
                        } else {
                            console.warn('[GeomPack Dual] iframe.contentWindow not available');
                        }
                    };

                    // Send message after iframe is loaded
                    if (iframeLoaded) {
                        sendMessage();
                    } else {
                        setTimeout(sendMessage, 500);
                    }
                };

                return r;
            };
        }
    }
});
