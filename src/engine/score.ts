import type { Engine } from './engine';
import type { Pillar } from './types';

/**
 * A generative score: a slow detuned drone whose chord follows the chapter's pillar, a soft
 * chime on every caption, all through a feedback-delay "hall". Pure WebAudio, no samples.
 */
const CHORDS: Record<Pillar, number[]> = {
  frame: [55.0, 82.41, 110.0, 164.81, 220.0], // A  E  A  E  A   — open fifths
  theory: [48.99, 73.42, 110.0, 146.83, 220.0], // G  D  A  D  A  — suspended, cool
  tools: [43.65, 65.41, 98.0, 130.81, 196.0], // F  C  G  C  G   — warmer
  data: [51.91, 77.78, 116.54, 155.56, 233.08], // G# D# A# D# A# — stranger
};

export class Score {
  private ctx: AudioContext;
  private master: GainNode;
  private voices: { osc: OscillatorNode; osc2: OscillatorNode; gain: GainNode }[] = [];
  private filter: BiquadFilterNode;
  private wet: GainNode;
  private muted = false;
  private level = 0.5;

  constructor() {
    const ctx = (this.ctx = new AudioContext());
    this.master = ctx.createGain();
    this.master.gain.value = 0;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -18;
    comp.ratio.value = 3;
    this.master.connect(comp).connect(ctx.destination);
    this.master.gain.linearRampToValueAtTime(this.level, ctx.currentTime + 4);

    // hall: two cross-fed delays with damping
    this.wet = ctx.createGain();
    this.wet.gain.value = 0.55;
    const dA = ctx.createDelay(1.5), dB = ctx.createDelay(1.5);
    dA.delayTime.value = 0.337;
    dB.delayTime.value = 0.521;
    const fA = ctx.createGain(), fB = ctx.createGain();
    fA.gain.value = 0.58;
    fB.gain.value = 0.54;
    const damp = ctx.createBiquadFilter();
    damp.type = 'lowpass';
    damp.frequency.value = 2400;
    this.wet.connect(dA);
    this.wet.connect(dB);
    dA.connect(fA).connect(damp).connect(dB);
    dB.connect(fB).connect(dA);
    dA.connect(this.master);
    dB.connect(this.master);

    this.filter = ctx.createBiquadFilter();
    this.filter.type = 'lowpass';
    this.filter.frequency.value = 520;
    this.filter.Q.value = 0.6;
    this.filter.connect(this.master);
    this.filter.connect(this.wet);

    // slow breathing of the filter
    const lfo = ctx.createOscillator(), lfoG = ctx.createGain();
    lfo.frequency.value = 0.045;
    lfoG.gain.value = 260;
    lfo.connect(lfoG).connect(this.filter.frequency);
    lfo.start();

    CHORDS.frame.forEach((f, i) => {
      const osc = ctx.createOscillator(), osc2 = ctx.createOscillator(), gain = ctx.createGain();
      osc.type = i < 2 ? 'sine' : 'triangle';
      osc2.type = 'sawtooth';
      osc.frequency.value = f;
      osc2.frequency.value = f * 1.004;
      osc2.detune.value = -7 + i * 3;
      const g2 = ctx.createGain();
      g2.gain.value = i < 2 ? 0.0 : 0.16;
      gain.gain.value = [0.5, 0.34, 0.2, 0.12, 0.07][i];
      osc.connect(gain);
      osc2.connect(g2).connect(gain);
      gain.connect(this.filter);
      osc.start();
      osc2.start();
      this.voices.push({ osc, osc2, gain });
    });
  }

  attach(engine: Engine) {
    engine.onChapter = (c) => this.chord(c.pillar);
    engine.onBeat = (c, b) => this.chime(c.pillar, b);
    const c = engine.chapters[Math.max(0, engine.active)];
    if (c) this.chord(c.pillar);
  }

  private chord(p: Pillar) {
    const t = this.ctx.currentTime;
    CHORDS[p].forEach((f, i) => {
      const v = this.voices[i];
      v.osc.frequency.cancelScheduledValues(t);
      v.osc2.frequency.cancelScheduledValues(t);
      v.osc.frequency.setTargetAtTime(f, t, 1.4);
      v.osc2.frequency.setTargetAtTime(f * 1.004, t, 1.4);
    });
    this.swell();
  }

  /** A low filtered swell at each chapter boundary. */
  private swell() {
    const ctx = this.ctx, t = ctx.currentTime;
    const n = ctx.createBufferSource();
    const buf = ctx.createBuffer(1, ctx.sampleRate * 3, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    n.buffer = buf;
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.Q.value = 1.2;
    bp.frequency.setValueAtTime(180, t);
    bp.frequency.exponentialRampToValueAtTime(1400, t + 2.6);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.1, t + 1.3);
    g.gain.linearRampToValueAtTime(0, t + 2.9);
    n.connect(bp).connect(g);
    g.connect(this.master);
    g.connect(this.wet);
    n.start(t);
    n.stop(t + 3);
  }

  private chime(p: Pillar, beat: number) {
    const ctx = this.ctx, t = ctx.currentTime;
    const base = CHORDS[p][4] * [1, 1.5, 2, 1.335, 2.25, 3][beat % 6];
    for (const [mult, amp] of [[1, 0.05], [2.01, 0.018]] as const) {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = 'sine';
      o.frequency.value = base * mult;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(amp, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 2.4);
      o.connect(g);
      g.connect(this.wet);
      g.connect(this.master);
      o.start(t);
      o.stop(t + 2.5);
    }
  }

  toggleMute() {
    this.muted = !this.muted;
    // The opening fade-in is a scheduled ramp. Left in place it would carry the volume back up after
    // a mute, so everything scheduled is dropped first and the gain continues from where it is.
    const g = this.master.gain, t = this.ctx.currentTime;
    if (g.cancelAndHoldAtTime) g.cancelAndHoldAtTime(t);
    else {
      const v = g.value;
      g.cancelScheduledValues(t);
      g.setValueAtTime(v, t);
    }
    g.setTargetAtTime(this.muted ? 0 : this.level, t, 0.2);
  }
}
