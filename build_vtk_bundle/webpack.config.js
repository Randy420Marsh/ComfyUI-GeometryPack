const path = require('path');

module.exports = {
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
};
