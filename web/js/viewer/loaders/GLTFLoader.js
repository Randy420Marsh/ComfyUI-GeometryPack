/**
 * GLTFLoader - GLTF/GLB mesh file loader with texture support
 *
 * Loads GLTF/GLB files using VTK.js vtkGLTFImporter.
 * Includes critical texture fixes for proper rendering:
 * - Removes metallicRoughnessTexture (causes rendering issues)
 * - Sets material to non-metallic for proper texture display
 * - Enables two-sided lighting
 *
 * This loader properly handles textured meshes, unlike simplified implementations.
 */

/**
 * Check if GLTFImporter is available
 * @param {Object} vtk - VTK.js global object
 * @returns {boolean}
 */
export function isGLTFSupported(vtk) {
    try {
        return vtk.IO.Geometry.vtkGLTFImporter !== undefined;
    } catch {
        return false;
    }
}

/**
 * Load a GLTF/GLB file with full texture support
 * @param {string} filepath - URL to the GLTF/GLB file
 * @param {Object} vtk - VTK.js global object
 * @param {Object} renderer - VTK renderer (required for GLTF import)
 * @param {Object} options - Loading options
 * @param {boolean} options.applyTextureFixes - Apply material fixes for proper texture display (default: true)
 * @param {boolean} options.twoSidedLighting - Enable two-sided lighting (default: true)
 * @returns {Object} Result object with actors and metadata
 */
export async function loadGLTF(filepath, vtk, renderer, options = {}) {
    const {
        applyTextureFixes = true,
        twoSidedLighting = true
    } = options;

    const vtkGLTFImporter = vtk.IO.Geometry.vtkGLTFImporter;

    if (!vtkGLTFImporter) {
        throw new Error('GLTFImporter not available in this VTK.js version');
    }

    // Enable two-sided lighting for better texture visibility
    if (twoSidedLighting && renderer.setTwoSidedLighting) {
        renderer.setTwoSidedLighting(true);
    }

    const importer = vtkGLTFImporter.newInstance();

    // CRITICAL: Set renderer FIRST - importActors() needs this
    importer.setRenderer(renderer);

    // Set the URL
    importer.setUrl(filepath);

    // Wait for GLTF parsing to complete using onReady callback
    await new Promise((resolve, reject) => {
        importer.onReady(() => {
            resolve();
        });

        // Start loading (this triggers parsing)
        importer.loadData().catch(reject);
    });

    // Import actors (creates VTK actors and adds them to renderer)
    importer.importActors();

    // Get actors from the importer (returns Map<string, vtkActor>)
    const actorsMap = importer.getActors();
    const actors = Array.from(actorsMap.values());

    // Track texture information
    let hasTexture = false;
    let hasMaterial = false;

    // Process actors and apply texture fixes
    if (applyTextureFixes) {
        actors.forEach((actor, index) => {
            const mapper = actor.getMapper();
            const property = actor.getProperty();
            const input = mapper?.getInputData();

            if (input && input.getNumberOfPoints() > 0) {
                if (property) {
                    // CRITICAL FIX: Remove metallicRoughnessTexture if present
                    // VTK.js GLTFImporter may use this instead of baseColorTexture
                    if (typeof property.setRMTexture === 'function') {
                        property.setRMTexture(null);
                    }
                    if (typeof property.setMetallicRoughnessTexture === 'function') {
                        property.setMetallicRoughnessTexture(null);
                    }

                    // Try multiple methods to get the base color texture
                    let texture = null;
                    if (typeof property.getBaseColorTexture === 'function') {
                        texture = property.getBaseColorTexture();
                    }
                    if (!texture && typeof property.getDiffuseTexture === 'function') {
                        texture = property.getDiffuseTexture();
                    }
                    if (!texture && typeof property.getTexture === 'function') {
                        texture = property.getTexture();
                    }

                    if (texture) {
                        hasTexture = true;
                        actor.addTexture(texture);
                    }

                    // Check for material
                    if (property.getMetallic !== undefined || property.getRoughness !== undefined) {
                        hasMaterial = true;
                    }

                    // Set material to non-metallic for proper texture display
                    if (typeof property.setMetallic === 'function') {
                        property.setMetallic(0.0);
                    }
                    if (typeof property.setRoughness === 'function') {
                        property.setRoughness(1.0);
                    }
                    if (typeof property.setInterpolation === 'function') {
                        property.setInterpolation(2); // Phong shading
                    }
                }

                // Configure mapper for texture rendering
                if (mapper) {
                    mapper.setScalarVisibility(false);
                    mapper.modified();
                }
            }
        });
    }

    // Calculate combined bounds from all actors
    let bounds = null;
    if (actors.length > 0) {
        bounds = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
        actors.forEach(actor => {
            const actorBounds = actor.getBounds();
            bounds[0] = Math.min(bounds[0], actorBounds[0]);
            bounds[1] = Math.max(bounds[1], actorBounds[1]);
            bounds[2] = Math.min(bounds[2], actorBounds[2]);
            bounds[3] = Math.max(bounds[3], actorBounds[3]);
            bounds[4] = Math.min(bounds[4], actorBounds[4]);
            bounds[5] = Math.max(bounds[5], actorBounds[5]);
        });
    }

    // Calculate total points/polys
    let totalPoints = 0;
    let totalPolys = 0;
    actors.forEach(actor => {
        const mapper = actor.getMapper();
        const input = mapper?.getInputData();
        if (input) {
            totalPoints += input.getNumberOfPoints();
            totalPolys += input.getNumberOfPolys();
        }
    });

    return {
        actors,
        importer,
        bounds,
        metadata: {
            format: 'GLTF',
            numActors: actors.length,
            numPoints: totalPoints,
            numPolys: totalPolys,
            bounds,
            hasTexture,
            hasMaterial
        }
    };
}

/**
 * Simplified GLTF load for non-textured display (e.g., field visualization)
 * Uses parseAsArrayBuffer instead of URL loading for simpler cases
 * @param {ArrayBuffer} arrayBuffer - File data as ArrayBuffer
 * @param {Object} vtk - VTK.js global object
 * @returns {Object} Result object with polyData and metadata
 */
export async function loadGLTFSimple(arrayBuffer, vtk) {
    const vtkGLTFImporter = vtk.IO.Geometry.vtkGLTFImporter;

    if (!vtkGLTFImporter) {
        throw new Error('GLTFImporter not available');
    }

    const importer = vtkGLTFImporter.newInstance();
    importer.parseAsArrayBuffer(arrayBuffer);
    await importer.update();

    const polyData = importer.getOutputData();
    if (!polyData) {
        throw new Error('Failed to extract polydata from GLTF');
    }

    return {
        polyData,
        importer,
        metadata: {
            format: 'GLTF',
            numPoints: polyData.getNumberOfPoints(),
            numPolys: polyData.getNumberOfPolys(),
            bounds: polyData.getBounds()
        }
    };
}

/**
 * Check if a filepath is a GLTF/GLB file
 * @param {string} filepath - File path or URL
 * @returns {boolean}
 */
export function canLoadGLTF(filepath) {
    const lower = filepath.toLowerCase();
    return lower.includes('.glb') || lower.includes('.gltf');
}

export default {
    load: loadGLTF,
    loadSimple: loadGLTFSimple,
    canLoad: canLoadGLTF,
    isSupported: isGLTFSupported,
    format: 'GLTF'
};
