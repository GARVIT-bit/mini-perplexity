// Group the 8 numbered passages by page.
//
// Why: retrieval often returns several chunks of the same page (for the csv
// question all 8 came from csv.html). Listing "csv — CSV file reading" eight
// times is noise; one card per page, listing its passage numbers, is what a
// reader wants.

/**
 * @param sources  [{n, url, title, text?}] in rank order
 * @param cited    passage numbers the answer actually used, first-use order
 * @returns [{url, title, numbers: [n], cited: boolean}]: cited pages first
 *          (in the order the answer first used them), then the rest by rank
 */
export function groupSources(sources, cited = []) {
  const byUrl = new Map();
  for (const s of sources) {
    if (!byUrl.has(s.url)) byUrl.set(s.url, { url: s.url, title: s.title, numbers: [], cited: false });
    byUrl.get(s.url).numbers.push(s.n);
  }
  const urlOf = new Map(sources.map((s) => [s.n, s.url]));
  const order = [];
  for (const n of cited) {
    const url = urlOf.get(n);
    if (url && !order.includes(url)) order.push(url);
  }
  for (const url of order) byUrl.get(url).cited = true;
  for (const url of byUrl.keys()) if (!order.includes(url)) order.push(url);
  return order.map((url) => byUrl.get(url));
}

/** Only http(s) links are clickable: a crawled "javascript:" URL must never become an href. */
export function safeHref(url) {
  try {
    const u = new URL(url);
    return u.protocol === 'http:' || u.protocol === 'https:' ? u.href : null;
  } catch {
    return null;
  }
}

/** "docs.python.org" from a URL, for compact source cards. */
export function hostOf(url) {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}
