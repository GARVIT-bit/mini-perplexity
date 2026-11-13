import { describe, expect, test } from 'vitest';
import { parseBlocks, parseInline } from './answerText.js';
import { groupSources, safeHref } from './sources.js';

describe('parseBlocks', () => {
  test('the real Gemini answer shape: paragraph, list, paragraph', () => {
    const text = 'Use a reader [1]. Steps:\n\n*   Open it [4][6].\n*   Loop [4].\n\nAlso DictReader [1].';
    expect(parseBlocks(text)).toEqual([
      { type: 'p', text: 'Use a reader [1]. Steps:' },
      { type: 'ul', items: ['Open it [4][6].', 'Loop [4].'] },
      { type: 'p', text: 'Also DictReader [1].' },
    ]);
  });

  test('code fences, headings, numbered lists', () => {
    expect(parseBlocks('## Answer\n1. one [1]\n2) two\n```\nx = 1\n```')).toEqual([
      { type: 'h', text: 'Answer' },
      { type: 'ul', items: ['one [1]', 'two'] },
      { type: 'code', text: 'x = 1' },
    ]);
  });

  test('an unclosed code fence mid-stream still shows its lines', () => {
    expect(parseBlocks('```\nimport csv')).toEqual([{ type: 'code', text: 'import csv' }]);
  });
});

describe('parseInline', () => {
  test('text, code, bold and every citation form', () => {
    expect(parseInline('Use `csv.reader` **now** [1][4] or [2, 3].')).toEqual([
      { type: 'text', value: 'Use ' },
      { type: 'code', value: 'csv.reader' },
      { type: 'text', value: ' ' },
      { type: 'bold', value: 'now' },
      { type: 'text', value: ' ' },
      { type: 'cite', nums: [1] },
      { type: 'cite', nums: [4] },
      { type: 'text', value: ' or ' },
      { type: 'cite', nums: [2, 3] },
      { type: 'text', value: '.' },
    ]);
  });

  test('a half-streamed citation stays text until its bracket closes', () => {
    expect(parseInline('Open it [4')).toEqual([{ type: 'text', value: 'Open it [4' }]);
  });

  test('brackets inside code are code, not citations', () => {
    expect(parseInline('`rows[0]`')).toEqual([{ type: 'code', value: 'rows[0]' }]);
  });
});

describe('groupSources', () => {
  const s = (n, url) => ({ n, url, title: url.toUpperCase() });
  const sources = [s(1, 'a'), s(2, 'b'), s(3, 'a'), s(4, 'c'), s(5, 'b')];

  test('one card per page, cited pages first in first-use order', () => {
    const g = groupSources(sources, [4, 3, 1]);
    expect(g.map((x) => [x.url, x.numbers, x.cited])).toEqual([
      ['c', [4], true],
      ['a', [1, 3], true],
      ['b', [2, 5], false],
    ]);
  });

  test('before the answer is done, rank order', () => {
    expect(groupSources(sources).map((x) => x.url)).toEqual(['a', 'b', 'c']);
  });

  test('a made-up citation number is ignored', () => {
    expect(groupSources(sources, [9]).every((x) => !x.cited)).toBe(true);
  });
});

test('safeHref only allows http(s)', () => {
  expect(safeHref('https://docs.python.org/3/')).toBe('https://docs.python.org/3/');
  expect(safeHref('javascript:alert(1)')).toBe(null);
  expect(safeHref('not a url')).toBe(null);
});
