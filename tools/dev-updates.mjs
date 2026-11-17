// Local coordinator controls: npm run dev:hold / dev:release / dev:updates [-- --port=4193].
import { parseArgs } from 'node:util';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { port: { type: 'string', default: '5199' } },
});
const action = positionals[0] ?? 'status';
if (!['hold', 'release', 'status'].includes(action)) throw new Error('Use hold, release or status');
const port = Number(values.port);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Use a port between 1 and 65535');
const response = await fetch(`http://127.0.0.1:${port}/__dev/updates${action === 'status' ? '' : '/' + action}`, { method: action === 'status' ? 'GET' : 'POST' });
if (!response.ok) throw new Error(`Development update gate returned ${response.status}`);
console.log(await response.text());
