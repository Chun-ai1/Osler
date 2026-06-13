// Treat .txt as a bundled asset so the vendored vis-network standalone build
// (assets/vendor/vis-network.min.txt) ships inside the app and Graph renders offline.
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('txt');

module.exports = config;
