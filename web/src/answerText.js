// Turn the LLM's answer text into blocks and inline pieces React can render.
//
// Why not a Markdown library + innerHTML: the text comes from an LLM that just
// read crawled web pages. Rendering it as HTML would let a page that says
// "<img src=x onerror=...>" run script in the user's browser. Here every piece
// becomes a React text node (escaped automatically) or one of our own elements.
//
// Supported, because the prompt asks for "a few sentences or a short list":
//   paragraphs, "- " / "* " / "1. " list items, ``` code blocks, # headings,
//   inline `code`, **bold**, and citations [3] / [1][4] / [1, 4].
// Anything else shows as plain text, which is a safe failure.

const LIST_ITEM = /^\s*(?:[-*+]|\d+[.)])\s+/;
const HEADING = /^\s*#{1,6}\s+/;
// one regex for every inline construct; the capture groups say which one matched
const INLINE = /`([^`]+)`|\*\*([^*]+)\*\*|\[(\d+(?:\s*,\s*\d+)*)\]/g;

/** Split answer text into blocks: {type: 'p'|'h'|'code', text} or {type: 'ul', items: [text]} */
export function parseBlocks(text) {
  const blocks = [];
  let para = [];
  let code = null; // lines of an open ``` block, or null
  const flushPara = () => {
    if (para.length) blocks.push({ type: 'p', text: para.join(' ') });
    para = [];
  };

  for (const line of text.split('\n')) {
    if (line.trim().startsWith('```')) {
      if (code) {
        blocks.push({ type: 'code', text: code.join('\n') });
        code = null;
      } else {
        flushPara();
        code = [];
      }
    } else if (code) {
      code.push(line);
    } else if (!line.trim()) {
      flushPara();
    } else if (HEADING.test(line)) {
      flushPara();
      blocks.push({ type: 'h', text: line.replace(HEADING, '') });
    } else if (LIST_ITEM.test(line)) {
      flushPara();
      const item = line.replace(LIST_ITEM, '');
      const last = blocks.at(-1);
      if (last?.type === 'ul') last.items.push(item);
      else blocks.push({ type: 'ul', items: [item] });
    } else {
      para.push(line.trim());
    }
  }
  if (code) blocks.push({ type: 'code', text: code.join('\n') }); // still streaming: show what arrived
  flushPara();
  return blocks;
}

/** Split one line into inline pieces: {type: 'text'|'code'|'bold', value} or {type: 'cite', nums: [n]} */
export function parseInline(text) {
  const out = [];
  let last = 0;
  for (const m of text.matchAll(INLINE)) {
    if (m.index > last) out.push({ type: 'text', value: text.slice(last, m.index) });
    if (m[1] !== undefined) out.push({ type: 'code', value: m[1] });
    else if (m[2] !== undefined) out.push({ type: 'bold', value: m[2] });
    else out.push({ type: 'cite', nums: m[3].split(',').map((n) => Number(n.trim())) });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ type: 'text', value: text.slice(last) });
  return out;
}
