/**
 * VTK.js Bundle with GLTFImporter support
 * This file bundles all required VTK.js modules into a single UMD file
 */

// Core rendering
import '@kitware/vtk.js/Rendering/Profiles/Geometry';

// Rendering
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkRenderWindow from '@kitware/vtk.js/Rendering/Core/RenderWindow';
import vtkRenderWindowInteractor from '@kitware/vtk.js/Rendering/Core/RenderWindowInteractor';
import vtkRenderer from '@kitware/vtk.js/Rendering/Core/Renderer';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkTexture from '@kitware/vtk.js/Rendering/Core/Texture';
import vtkProperty from '@kitware/vtk.js/Rendering/Core/Property';
import vtkAnnotatedCubeActor from '@kitware/vtk.js/Rendering/Core/AnnotatedCubeActor';
import vtkCamera from '@kitware/vtk.js/Rendering/Core/Camera';

// OpenGL implementations
import vtkOpenGLRenderWindow from '@kitware/vtk.js/Rendering/OpenGL/RenderWindow';

// Interaction
import vtkInteractorStyleTrackballCamera from '@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera';
import vtkOrientationMarkerWidget from '@kitware/vtk.js/Interaction/Widgets/OrientationMarkerWidget';

// IO - Geometry readers
import vtkSTLReader from '@kitware/vtk.js/IO/Geometry/STLReader';
import vtkPLYReader from '@kitware/vtk.js/IO/Geometry/PLYReader';

// IO - Misc readers
import vtkOBJReader from '@kitware/vtk.js/IO/Misc/OBJReader';

// IO - XML readers
import vtkXMLPolyDataReader from '@kitware/vtk.js/IO/XML/XMLPolyDataReader';

// IO - GLTF (the key addition!)
import vtkGLTFImporter from '@kitware/vtk.js/IO/Geometry/GLTFImporter';
const gltfAvailable = true;
console.log('[VTK Bundle] GLTFImporter included in bundle');

// Filters
import vtkPolyDataNormals from '@kitware/vtk.js/Filters/Core/PolyDataNormals';

// Sources
import vtkConeSource from '@kitware/vtk.js/Filters/Sources/ConeSource';
import vtkSphereSource from '@kitware/vtk.js/Filters/Sources/SphereSource';
import vtkCubeSource from '@kitware/vtk.js/Filters/Sources/CubeSource';

// Color maps
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
// Note: ColorMaps import removed - causes internal VTK.js initialization error

// Export everything as a global vtk object (UMD style)
const vtk = {
    Rendering: {
        Core: {
            vtkActor,
            vtkMapper,
            vtkTexture,
            vtkProperty,
            vtkAnnotatedCubeActor,
            vtkCamera,
            vtkRenderWindow,
            vtkRenderWindowInteractor,
            vtkRenderer,
            vtkColorTransferFunction,
        },
        Misc: {
            vtkFullScreenRenderWindow,
        },
        OpenGL: {
            vtkRenderWindow: vtkOpenGLRenderWindow,
        },
    },
    Interaction: {
        Style: {
            vtkInteractorStyleTrackballCamera,
        },
        Widgets: {
            vtkOrientationMarkerWidget,
        },
    },
    IO: {
        Geometry: {
            vtkSTLReader,
            vtkPLYReader,
            vtkGLTFImporter, // Will be null if not available
        },
        Misc: {
            vtkOBJReader,
        },
        XML: {
            vtkXMLPolyDataReader,
        },
    },
    Filters: {
        Core: {
            vtkPolyDataNormals,
        },
        Sources: {
            vtkConeSource,
            vtkSphereSource,
            vtkCubeSource,
        },
    },
    // Metadata
    _gltfAvailable: gltfAvailable,
};

// Export for UMD
if (typeof module !== 'undefined' && module.exports) {
    module.exports = vtk;
}
if (typeof window !== 'undefined') {
    window.vtk = vtk;
}
if (typeof global !== 'undefined') {
    global.vtk = vtk;
}

export default vtk;
