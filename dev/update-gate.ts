import { createHash } from 'node:crypto';
import { existsSync, unlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { Plugin, ViteDevServer } from 'vite';

/** Server-only: hold updates during a coordinated editing pass, then reload readers once. */
export function updateGate(): Plugin {
  let server: ViteDevServer;
  let held = false, dirty = false;
  let gateFile = '';
  let timer: ReturnType<typeof setTimeout> | undefined;
  const quietMs = 900;
  const cancel = () => { clearTimeout(timer); timer = undefined; };
  const flush = () => {
    timer = undefined;
    if (held || !dirty) return;
    dirty = false;
    // Invalidate every changed dependency before one coherent client reload. No old transformed
    // module may survive changes held back from Vite's normal HMR propagation.
    for (const environment of Object.values(server.environments)) environment.moduleGraph.invalidateAll();
    server.environments.client.hot.send({ type: 'full-reload', path: '*' });
    server.config.logger.info('[updates] refreshed readers after the editing pass');
  };
  const schedule = () => { cancel(); if (!held && dirty) timer = setTimeout(flush, quietMs); };
  const status = () => ({ held, dirty, quietMs });
  return {
    name: 'piramid-development-update-gate',
    apply: 'serve',
    config() {
      return {
        server: {
          // Vite 8's server-side HMR scheduling API. Holding the task here also covers HTML
          // and CSS, which can otherwise bypass a module-only hotUpdate hook.
          hotUpdateEnvironments: async () => {},
        },
      };
    },
    hotUpdate(options) {
      const root = server.config.root;
      const publicDir = server.config.publicDir;
      // A server may have restarted while its reader remains open. Its module graph is then
      // empty until the next request, but edits to reader sources/assets still need a refresh.
      const readerFile = options.file.startsWith(`${root}/src/`)
        || (!!publicDir && options.file.startsWith(`${publicDir}/`))
        || (options.file.startsWith(`${root}/`) && !options.file.slice(root.length + 1).includes('/') && options.file.endsWith('.html'));
      if (this.environment.name === 'client' && (options.modules.length || readerFile)) {
        dirty = true;
        schedule();
      }
      return [];
    },
    configureServer(devServer) {
      server = devServer;
      const scope = createHash('sha256').update(server.config.root).digest('hex').slice(0, 12);
      gateFile = join(tmpdir(), `piramid-dev-updates-${scope}.hold`);
      held = existsSync(gateFile);
      server.config.logger.info(`[updates] ${held ? 'held; release with npm run dev:release' : 'automatic; batch with npm run dev:hold'}`);
      server.middlewares.use('/__dev/updates', (req, res, next) => {
        const address = req.socket.remoteAddress;
        if (!['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(address ?? '')) { res.statusCode = 403; res.end(); return; }
        const origin = req.headers.origin;
        if (origin && origin !== `http://${req.headers.host}`) { res.statusCode = 403; res.end(); return; }
        const action = new URL(req.url ?? '/', 'http://localhost').pathname;
        if (req.method === 'POST' && action === '/hold') {
          held = true; cancel(); writeFileSync(gateFile, 'held\n');
          server.config.logger.info('[updates] held; readers stay on their current slide');
        } else if (req.method === 'POST' && action === '/release') {
          held = false;
          if (existsSync(gateFile)) unlinkSync(gateFile);
          schedule();
          server.config.logger.info('[updates] released; pending changes refresh once after saving settles');
        } else if (req.method !== 'GET' || (action !== '/' && action !== '')) { next(); return; }
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Cache-Control', 'no-store');
        res.end(JSON.stringify(status()));
      });
      server.httpServer?.once('close', cancel);
    },
  };
}
