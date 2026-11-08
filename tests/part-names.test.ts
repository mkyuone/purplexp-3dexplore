import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { CadOccurrence } from '../src/cad.ts'
import { collectPartNameRows, exportPartNames, importPartNames, type PartNameRow } from '../src/part-names.ts'

const rows: readonly PartNameRow[] = [
  { id: '0:1/0:2', stepName: 'Wheel', displayName: 'Existing left', assemblyPath: 'Chassis' },
  { id: '0:1/0:3', stepName: 'Wheel', displayName: 'Existing right', assemblyPath: 'Chassis' },
  { id: '0:4', stepName: 'Body', displayName: 'Body' },
]
const header = 'Display Name,STEP Name,Assembly Path,Part ID'

test('exports a UTF-8 spreadsheet CSV and round-trips Unicode, quotes and embedded line breaks', () => {
  const part = { id: '0:1/0:2', stepName: 'Wheel, "left"', displayName: 'Étrier, "avant"\r\n左側  ', assemblyPath: 'Body / Front, left' }
  const csv = exportPartNames([part])
  assert.ok(csv.startsWith(`\uFEFF${header}\r\n`))
  assert.match(csv, /"Étrier, ""avant""\r\n左側  "/)
  assert.match(csv, /"part:0:1\/0:2"\r\n$/)
  assert.deepEqual(importPartNames(csv, [{ ...part, displayName: '' }]), { [part.id]: part.displayName })
})

test('imports only provided rows and represents blank display names as explicit clears', () => {
  const csv = `${header}\nRight wheel,Wheel,Chassis,part:0:1/0:3\n"  ",Wheel,Chassis,part:0:1/0:2\n\n`
  assert.deepEqual(importPartNames(csv, rows), { '0:1/0:3': 'Right wheel', '0:1/0:2': '' })
  assert.deepEqual(importPartNames(header, rows), {})
  assert.equal(rows[0].displayName, 'Existing left')
})

test('does not retain a redundant override equal to the original STEP name', () => {
  assert.deepEqual(importPartNames(`${header}\nWheel,Wheel,Chassis,part:0:1/0:2`, rows), { '0:1/0:2': '' })
})

test('accepts semicolon and tab separators with reordered spreadsheet columns', () => {
  for (const separator of [';', '\t']) {
    const columns = ['Part ID', 'Assembly Path', 'STEP Name', 'Display Name'].map(cell => `"${cell}"`)
    const data = ['part:0:1/0:2', 'Chassis', 'Wheel', `"Front${separator} left"`]
    const csv = [columns.join(separator), data.join(separator)].join('\r\n')
    assert.deepEqual(importPartNames(csv, rows), { '0:1/0:2': `Front${separator} left` })
  }
})

test('rejects invalid edits before returning any updates', () => {
  const valid = 'Changed,Wheel,Chassis,part:0:1/0:2'
  const invalid = [
    [valid, /repeats a Part ID/],
    ['Name,Wheel,Chassis,part:0:8', /unknown Part ID/],
    ['Name,Wheel,Chassis,0:1/0:3', /unknown Part ID/],
    ['Name,Axle,Chassis,part:0:1/0:3', /does not match/],
    ['Name,Wheel,Another assembly,part:0:1/0:3', /does not match/],
    ['Name,Wheel,Chassis', /must have 4 columns/],
    ['"Name,Wheel,Chassis,part:0:1/0:3', /invalid quoting/],
    ['"Name"tail,Wheel,Chassis,part:0:1/0:3', /invalid quoting/],
    ['Nam"e,Wheel,Chassis,part:0:1/0:3', /invalid quoting/],
  ] as const
  for (const [line, message] of invalid) {
    assert.throws(() => importPartNames(`${header}\n${valid}\n${line}`, rows), message)
  }
  assert.throws(() => importPartNames('Display Name,STEP Name,STEP Name,Part ID', rows), /CSV headers/)
  assert.equal(rows[0].displayName, 'Existing left')
})

test('neutralizes formula-looking cells across columns and preserves original text on re-import', () => {
  const parts = [
    { id: '1', stepName: '+123', displayName: '=HYPERLINK("https://example.org", "Open")', assemblyPath: '@Drive' },
    { id: '2', stepName: 'Wheel', displayName: '  ＝1+2', assemblyPath: '\t-Base' },
    { id: '3', stepName: 'text:-Beam', displayName: 'text:=literal' },
  ]
  const csv = exportPartNames(parts)
  assert.match(csv, /"text:=HYPERLINK\(""https:\/\/example.org"", ""Open""\)","text:\+123","text:@Drive","part:1"/)
  assert.match(csv, /"text:  ＝1\+2","Wheel","text:\t-Base"/)
  assert.match(csv, /"text:text:=literal","text:text:-Beam"/)
  assert.deepEqual(importPartNames(csv, parts.map(part => ({ ...part, displayName: '' }))),
    Object.fromEntries(parts.map(part => [part.id, part.displayName])))
})

test('keeps number-like, date-like and Boolean-looking labels as spreadsheet text', () => {
  const part = { id: '0:1/0:2', stepName: '2026-11-08', displayName: '0012', assemblyPath: 'false' }
  const csv = exportPartNames([part])
  assert.match(csv, /"text:0012","text:2026-11-08","text:false","part:0:1\/0:2"/)
  assert.deepEqual(importPartNames(csv, [part]), { [part.id]: part.displayName })
})

test('collects visible occurrences in tree order using original assembly names', () => {
  const part = (id: string, name: string, mesh?: number, children: CadOccurrence[] = []): CadOccurrence =>
    ({ id, name, mesh, children, matrix: [] })
  const roots = [part('root', 'Original Root', undefined, [
    part('empty', 'Metadata', undefined, [part('empty-child', 'Empty child')]),
    part('assembly', 'Original Assembly', undefined, [part('wheel', 'Wheel', 0)]),
    part('other-wheel', 'Wheel', 0),
  ])]
  assert.deepEqual(collectPartNameRows(roots, { root: 'Renamed root', wheel: 'Front wheel' }), [
    { id: 'root', stepName: 'Original Root', displayName: 'Renamed root', assemblyPath: '' },
    { id: 'assembly', stepName: 'Original Assembly', displayName: '', assemblyPath: 'Original Root' },
    { id: 'wheel', stepName: 'Wheel', displayName: 'Front wheel', assemblyPath: 'Original Root / Original Assembly' },
    { id: 'other-wheel', stepName: 'Wheel', displayName: '', assemblyPath: 'Original Root' },
  ])
})
