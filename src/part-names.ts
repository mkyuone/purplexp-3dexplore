import type { CadOccurrence } from './cad.ts'

export interface PartNameRow {
  id: string
  stepName: string
  displayName: string
  assemblyPath?: string
}

const columns = ['Display Name', 'STEP Name', 'Assembly Path', 'Part ID'] as const
const partPrefix = 'part:'
const textPrefix = 'text:'
const formulaStart = /^[\t\r\n]|^[\s'\u200b-\u200d\u2060]*[=+\-@＝＋－＠]/u
const spreadsheetScalar = /^(?:(?:\p{Sc} ?)?(?:\p{Nd}[\p{Nd}.,:/ -]*|[.,]\p{Nd}+)(?:[eE][+-]?\p{Nd}+)?(?: ?[%‰\p{Sc}])?|true|false)$/iu

export function collectPartNameRows(
  roots: readonly CadOccurrence[], overrides: Readonly<Record<string, string>>,
): PartNameRow[] {
  const rows: PartNameRow[] = []
  function visit(node: CadOccurrence, ancestors: string[]) {
    const start = rows.length
    const override = Object.hasOwn(overrides, node.id) ? overrides[node.id] : ''
    rows.push({ id: node.id, stepName: node.name, displayName: override, assemblyPath: ancestors.join(' / ') })
    const path = [...ancestors, node.name]
    for (const child of node.children) visit(child, path)
    if (node.mesh === undefined && rows.length === start + 1) rows.pop()
  }
  for (const root of roots) visit(root, [])
  return rows
}

function needsTextPrefix(value: string) {
  return value.startsWith(textPrefix) || value.startsWith("'") || formulaStart.test(value) || spreadsheetScalar.test(value.trim())
}

function spreadsheetText(value: string) {
  // An ordinary text prefix survives a spreadsheet save without depending on
  // whether that application strips an apostrophe or tab used as an escape.
  return needsTextPrefix(value) ? textPrefix + value : value
}

function readSpreadsheetText(value: string) {
  const unescaped = value.slice(textPrefix.length)
  return value.startsWith(textPrefix) && needsTextPrefix(unescaped) ? unescaped : value
}

function quoteCell(value: string) {
  return `"${value.replaceAll('"', '""')}"`
}

export function exportPartNames(rows: readonly PartNameRow[]): string {
  const lines = [columns.join(',')]
  for (const row of rows) {
    lines.push([
      spreadsheetText(row.displayName), spreadsheetText(row.stepName), spreadsheetText(row.assemblyPath ?? ''),
      partPrefix + row.id,
    ].map(quoteCell).join(','))
  }
  return `\uFEFF${lines.join('\r\n')}\r\n`
}

function readTable(text: string, separator: string): string[][] {
  const records: string[][] = []
  let fields: string[] = []
  let field = ''
  let state: 'start' | 'plain' | 'quoted' | 'closed' = 'start'
  const malformed = () => new Error(`CSV row ${records.length + 1} has invalid quoting.`)

  for (let index = 0; index < text.length; index++) {
    const character = text[index]
    if (state === 'quoted') {
      if (character === '"' && text[index + 1] === '"') {
        field += '"'
        index++
      } else if (character === '"') state = 'closed'
      else field += character
      continue
    }

    if (character === separator || character === '\n' || character === '\r') {
      fields.push(field)
      if (character !== separator) {
        records.push(fields)
        fields = []
        if (character === '\r' && text[index + 1] === '\n') index++
      }
      field = ''
      state = 'start'
    } else if (state === 'closed') {
      throw malformed()
    } else if (character === '"') {
      if (state !== 'start') throw malformed()
      state = 'quoted'
    } else {
      field += character
      state = 'plain'
    }
  }
  if (state === 'quoted') throw malformed()
  if (fields.length || state !== 'start') records.push([...fields, field])
  return records
}

function readHeader(text: string) {
  const line = text.split(/[\r\n]/, 1)[0]
  for (const separator of [',', ';', '\t']) {
    try {
      const [header] = readTable(line, separator)
      if (header?.length === columns.length && columns.every(column => header.includes(column))) {
        return { separator, header }
      }
    } catch {
      // A quoted header can be malformed with the wrong field separator.
    }
  }
  throw new Error(`CSV headers must include ${columns.join(', ')}.`)
}

function sameLabel(left: string, right: string) {
  return left.replace(/\r\n?/g, '\n') === right.replace(/\r\n?/g, '\n')
}

/** Return only rows supplied by the CSV. An empty value clears that override. */
export function importPartNames(text: string, rows: readonly PartNameRow[]): Record<string, string> {
  const csv = text.replace(/^\uFEFF/, '')
  if (csv.includes('\0')) throw new Error('The file is not valid CSV text.')
  const { separator, header } = readHeader(csv)
  const [display, step, assembly, part] = columns.map(column => header.indexOf(column))
  const known = new Map(rows.map(row => [partPrefix + row.id, row]))
  const updates = new Map<string, string>()
  const [, ...records] = readTable(csv, separator)
  for (const [index, fields] of records.entries()) {
    if (fields.every(field => field.trim() === '')) continue
    const rowNumber = index + 2
    if (fields.length !== columns.length) throw new Error(`CSV row ${rowNumber} must have ${columns.length} columns.`)
    const row = known.get(fields[part])
    if (!row) throw new Error(`CSV row ${rowNumber} has an unknown Part ID.`)
    if (updates.has(row.id)) throw new Error(`CSV row ${rowNumber} repeats a Part ID.`)
    if (!sameLabel(readSpreadsheetText(fields[step]), row.stepName)
        || !sameLabel(readSpreadsheetText(fields[assembly]), row.assemblyPath ?? '')) {
      throw new Error(`CSV row ${rowNumber} does not match its STEP name or assembly path.`)
    }
    const name = readSpreadsheetText(fields[display])
    updates.set(row.id, name.trim() && !sameLabel(name, row.stepName) ? name : '')
  }
  return Object.fromEntries(updates)
}
