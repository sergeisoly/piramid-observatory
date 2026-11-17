// The capture and diagnostic tools accept --key=value and treat a bare flag as '1'.
export function readFlags() {
  return Object.fromEntries(process.argv.slice(2).map((arg) => {
    const [key, ...value] = arg.replace(/^--/, '').split('=');
    return [key, value.join('=') || '1'];
  }));
}
