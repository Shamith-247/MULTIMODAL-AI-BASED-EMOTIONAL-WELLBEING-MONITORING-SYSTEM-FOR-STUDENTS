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
  // Handle POST endpoint for automated evaluation reporting
  if (req.method === 'POST' && req.url === '/api/test-results') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const outPath = path.join(BASE_DIR, 'sample_evaluation_results.json');
        fs.writeFileSync(outPath, body, 'utf8');
        console.log('\n[EVALUATION REPORT RECEIVED] Successfully written to:', outPath);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'ok', writtenTo: outPath }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  let reqPath = decodeURI(req.url.split('?')[0]);
  if (reqPath === '/' || reqPath === '') reqPath = '/index.html';

  const ROOT_DIR = path.resolve(__dirname, '..');
  let filePath;

  if (reqPath.startsWith('/models/')) {
    // Check local app/models first (where face-api shards reside)
    const appModelsPath = path.join(BASE_DIR, reqPath);
    if (fs.existsSync(appModelsPath)) {
      filePath = appModelsPath;
    } else {
      filePath = path.join(ROOT_DIR, reqPath);
    }
    if (!filePath.startsWith(ROOT_DIR) && !filePath.startsWith(BASE_DIR)) {
      res.writeHead(403);
      res.end('Access denied');
      return;
    }
  } else if (reqPath.startsWith('/text_model/') || reqPath.startsWith('/face_model/') || reqPath.startsWith('/voice_model/')) {
    filePath = path.join(ROOT_DIR, reqPath);
    if (!filePath.startsWith(ROOT_DIR)) {
      res.writeHead(403);
      res.end('Access denied');
      return;
    }
  } else {
    filePath = path.join(BASE_DIR, reqPath);
    if (!filePath.startsWith(BASE_DIR)) {
      res.writeHead(403);
      res.end('Access denied');
      return;
    }
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
  console.log(`\n  1. Unified Multimodal Model Tester (Face + Text + Fusion):`);
  console.log(`     --> http://localhost:${PORT}/test_multimodal_models.html`);
  console.log(`\n  2. Situational Scenario Emotion Tester (Text):`);
  console.log(`     --> http://localhost:${PORT}/test_text_scenarios.html`);
  console.log(`\n  3. Dedicated Webcam & Face Output Tester:`);
  console.log(`     --> http://localhost:${PORT}/test_webcam_face.html`);
  console.log(`\n  4. Full SignalWell Multimodal Console:`);
  console.log(`     --> http://localhost:${PORT}/index.html`);
  console.log(`\n  Press Ctrl + C to stop the server.\n`);
  console.log('='.repeat(68) + '\n');
});

