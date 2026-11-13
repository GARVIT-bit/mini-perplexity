// Incremental Server-Sent Events parser.
//
// Why incremental: network chunks don't line up with events. One read can end
// in the middle of an event ("event: tok") and the next read finishes it. So we
// keep a buffer, cut off every complete event (they end with a blank line,
// "\n\n"), and keep the unfinished tail for the next chunk.
//
// Each parsed event keeps its exact raw text too, so the proxy can forward it
// byte for byte instead of re-serializing (nothing can get lost in translation).

export class SseParser {
  constructor() {
    this.buffer = '';
  }

  /** Feed a decoded text chunk; returns the events it completed: [{event, data, raw}] */
  push(text) {
    // normalize CRLF so a Windows-style server can't break the "\n\n" split
    this.buffer += text.replace(/\r\n/g, '\n');
    const events = [];
    let end;
    while ((end = this.buffer.indexOf('\n\n')) !== -1) {
      const block = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end + 2);
      const parsed = parseBlock(block);
      if (parsed) events.push(parsed);
    }
    return events;
  }
}

function parseBlock(block) {
  let event = 'message'; // the SSE default when no "event:" line is present
  const dataLines = [];
  for (const line of block.split('\n')) {
    if (!line || line.startsWith(':')) continue; // ":" lines are comments / keep-alives
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') event = value;
    else if (field === 'data') dataLines.push(value);
  }
  if (!dataLines.length) return null;
  const rawData = dataLines.join('\n');
  let data;
  try {
    data = JSON.parse(rawData); // our Python service always sends JSON
  } catch {
    data = rawData;
  }
  return { event, data, raw: `${block}\n\n` };
}

/** Format one event the same way the Python service does. */
export function sse(event, data) {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}
