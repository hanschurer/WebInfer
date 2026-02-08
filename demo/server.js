/**
 * Simple development server for testing WebInfer.js
 * 
 * Usage: node demo/server.js
 */

import { createServer } from 'http';
import { readFile } from 'fs/promises';
import { extname, join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = join(__dirname, '..');

const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
};

const PORT = process.env.PORT || 3000;

const server = createServer(async (req, res) => {
  let url = req.url || '/';
  
  // Default to demo/index.html
  if (url === '/') {
    url = '/demo/index.html';
  }

  const filePath = join(ROOT, url);
  const ext = extname(filePath);
  const mimeType = MIME_TYPES[ext] || 'application/octet-stream';

  try {
    const content = await readFile(filePath);
    
    // Add CORS and security headers for WebGPU/WASM
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Content-Type', mimeType);
    res.setHeader('Access-Control-Allow-Origin', '*');
    
    res.writeHead(200);
    res.end(content);
  } catch (error) {
    if (error.code === 'ENOENT') {
      res.writeHead(404);
      res.end(`File not found: ${url}`);
    } else {
      res.writeHead(500);
      res.end(`Server error: ${error.message}`);
    }
  }
});

server.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════════╗
║                                                      ║
║   ⚡ WebInfer.js Development Server                  ║
║                                                      ║
║   Local:   http://localhost:${PORT}                     ║
║                                                      ║
║   Press Ctrl+C to stop                               ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
`);
});
