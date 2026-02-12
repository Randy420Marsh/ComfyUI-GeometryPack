const path = require('path');

module.exports = [
    // Existing VTK.js bundle (unchanged)
    {
        entry: './vtk_gltf_bundle.js',
        output: {
            filename: 'vtk-gltf.js',
            path: path.resolve(__dirname, '../web/js'),  // Output to web/js directory
            library: {
                name: 'vtk',
                type: 'umd',
                export: 'default',
            },
            globalObject: 'this',
        },
        mode: 'production',
        resolve: {
            extensions: ['.js'],
        },
        module: {
            rules: [
                {
                    test: /\.js$/,
                    exclude: /node_modules/,
                    use: {
                        loader: 'babel-loader',
                        options: {
                            presets: ['@babel/preset-env'],
                        },
                    },
                },
            ],
        },
        // Optimize for size
        optimization: {
            minimize: true,
        },
    },
    // New viewer modules bundle
    {
        entry: path.resolve(__dirname, '../web/js/viewer/index.js'),
        output: {
            filename: 'viewer-bundle.js',
            path: path.resolve(__dirname, '../web/js'),
            library: {
                name: 'GeomPackViewer',
                type: 'umd',
            },
            globalObject: 'this',
        },
        mode: 'production',
        resolve: {
            extensions: ['.js'],
        },
        module: {
            rules: [
                {
                    test: /\.js$/,
                    exclude: /node_modules/,
                    use: {
                        loader: 'babel-loader',
                        options: {
                            presets: ['@babel/preset-env'],
                        },
                    },
                },
                {
                    test: /\.css$/,
                    use: ['style-loader', 'css-loader'],
                },
            ],
        },
        optimization: {
            minimize: true,
        },
    },
];
