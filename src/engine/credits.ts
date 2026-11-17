/**
 * Who created the presentation. It is independent, and a viewer should be able to tell at a glance
 * that it speaks for its author and not for PIRAMID: the poster and the end card both name the author.
 * Edit the two values to change the byline.
 */
export const AUTHOR = { name: 'Sergei Solovev', url: 'https://sergeisoly.github.io/' };

/**
 * Optional repository and contact links. The end card displays the entries the owner fills in.
 */
export const CODE_URL: string = 'https://github.com/sergeisoly/piramid-observatory';
export const CONTACT: { label: string; url: string } = { label: '', url: '' };

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
const link = (href: string, text: string) => `<a href="${esc(href)}" target="_blank" rel="noopener noreferrer">${esc(text)}</a>`;
/** "The code: … · Contact: …", or an empty string when neither is set. */
export const REACH_HTML = [
  CODE_URL ? `The code: ${link(CODE_URL, CODE_URL.replace(/^https?:\/\//, ''))}` : '',
  CONTACT.url ? `Contact: ${link(CONTACT.url, CONTACT.label || CONTACT.url)}` : '',
].filter(Boolean).join(' · ');

export const CREDIT_HTML = `<a href="${AUTHOR.url}" target="_blank" rel="noopener noreferrer"><b>${AUTHOR.name}</b></a> created this independent presentation of PIRAMID’s research, including its simulations and synthesis. The authors of the cited research have not reviewed it.`;

/** Compact credit for the opening poster; the end card keeps its fuller disclosure. */
export const POSTER_CREDIT_HTML = `This presentation by <b>${AUTHOR.name}</b> is independent of PIRAMID and was not reviewed by the cited researchers.`;
