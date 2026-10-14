#!/usr/bin/env node
/**
 * SignalWell - Lightweight Local Development Server
 * ==================================================
 * Serves the SignalWell web app and webcam diagnostic tester locally
 * with correct MIME types for binary weights (.shard, .json, .onnx).
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3000;
const BASE_DIR = path.join(__dirname, 'app');

const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.onnx': 'application/octet-stream',
  '.shard1': 'application/octet-stream',
  '.shard2': 'application/octet-stream'
};

const server = http.createServer((req, res) => {
  let reqPath = decodeURI(req.url.split('?')[0]);
  if (reqPath === '/' || reqPath === '') reqPath = '/index.html';

  let filePath = path.join(BASE_DIR, reqPath);
  
  // Security check: prevent directory traversal outside BASE_DIR
  if (!filePath.startsWith(BASE_DIR)) {
    res.writeHead(403);
    res.end('Access denied');
    return;
  }

  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('File Not Found: ' + reqPath);
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    res.writeHead(200, {
      'Content-Type': contentType,
      'Access-Control-Allow-Origin': '*',
      'Cache-Control': 'no-cache'
    });

    const stream = fs.createReadStream(filePath);
    stream.pipe(res);
  });
});

server.listen(PORT, () => {
  console.log('\n' + '='.repeat(68));
  console.log(' SignalWell Local Web Server Running');
  console.log('='.repeat(68));
  console.log(`\n  1. Situational Scenario Emotion Tester (Text):`);
  console.log(`     --> http://localhost:${PORT}/test_text_scenarios.html`);
  console.log(`\n  2. Dedicated Webcam & Face Output Tester:`);
  console.log(`     --> http://localhost:${PORT}/test_webcam_face.html`);
  console.log(`\n  3. Full SignalWell Multimodal Console:`);
  console.log(`     --> http://localhost:${PORT}/index.html`);
  console.log(`\n  Press Ctrl + C to stop the server.\n`);
  console.log('='.repeat(68) + '\n');
});

