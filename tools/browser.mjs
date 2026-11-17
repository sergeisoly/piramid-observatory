import puppeteer from 'puppeteer-core';

// Keep the WebGPU launch flags consistent across the browser checks and capture tools.
export function launchChrome({
  chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headful = false,
  width = 1600,
  height = 900,
  windowArgs = [`--window-size=${width},${height}`, '--hide-scrollbars'],
  extraArgs = [],
} = {}) {
  return puppeteer.launch({
    executablePath: chrome,
    headless: !headful,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=WebGPU',
      '--use-angle=metal',
      '--enable-gpu',
      '--ignore-gpu-blocklist',
      ...windowArgs,
      '--mute-audio',
      ...extraArgs,
    ],
    defaultViewport: { width, height, deviceScaleFactor: 1 },
  });
}
