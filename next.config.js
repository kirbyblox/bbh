/** @type {import('next').NextConfig} */
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  reactStrictMode: true,
  //distDir: 'build',
  webpack: (config, { }) => {

    config.resolve.extensions.push(".ts", ".tsx");
    config.resolve.fallback = { fs: false };
    config.optimization.minimize = false;
    mode: 'development',
    config.plugins.push(
    new CopyPlugin({
      patterns: [
        {
          from: './node_modules/onnxruntime-web/dist/ort-wasm.wasm',
          to: 'static/chunks/pages',
        },             {
          from: './node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm',
          to: 'static/chunks/pages',
        },          
          {
            from: './model',
            to: 'static/chunks/pages',
          },
        ],
      }),
    );

    return config;
  } 
}
