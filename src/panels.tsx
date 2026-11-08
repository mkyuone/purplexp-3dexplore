import { Box, ChevronRight, Download, Layers3, RotateCcw, Undo2, Upload } from 'lucide-react';
import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import type { CadOccurrence } from './cad';

export function indexOccurrences(roots: CadOccurrence[]) {
  const byId = new Map<string, CadOccurrence>();
  const parents = new Map<string, string>();
  let parts = 0;
  const visible = new Set<string>();
  function visit(node: CadOccurrence, parent?: string): boolean {
    byId.set(node.id, node);
    if (parent !== undefined) parents.set(node.id, parent);
    const hasMesh = node.mesh !== undefined;
    if (hasMesh) parts++;
    const children = node.children.map(child => visit(child, node.id));
    if (!hasMesh && !children.includes(true)) return false;
    visible.add(node.id);
    return true;
  }
  roots.forEach(node => visit(node));
  return { byId, parents, parts, visible };
}

type AssemblyIndex = ReturnType<typeof indexOccurrences>;

export function AssemblyPanel({ roots, index, selected, onSelect, displayName, onExportNames, onImportNames, namesMessage }: {
  roots: CadOccurrence[];
  index: AssemblyIndex;
  selected: string | null;
  onSelect: (id: string) => void;
  displayName?: (node: CadOccurrence) => string;
  onExportNames?: () => void;
  onImportNames?: (file: File) => void;
  namesMessage?: { text: string; error?: boolean } | null;
}) {
  const [expanded, setExpanded] = useState(() => new Set(roots.map(root => root.id)));
  const namesPicker = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!selected) return;
    const ancestors: string[] = [];
    for (let parent = index.parents.get(selected); parent; parent = index.parents.get(parent)) {
      ancestors.push(parent);
    }
    setExpanded(previous => ancestors.every(id => previous.has(id)) ? previous : new Set([...previous, ...ancestors]));
  }, [index, selected]);

  function toggle(id: string) {
    setExpanded(previous => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function renderNode(node: CadOccurrence, depth: number): ReactNode {
    if (!index.visible.has(node.id)) return null;
    const group = node.children.some(child => index.visible.has(child.id));
    const open = expanded.has(node.id);
    const name = displayName?.(node) ?? node.name;
    return <li key={node.id}>
      <div className={`assembly-row${selected === node.id ? ' is-selected' : ''}`} style={{ paddingInlineStart: 4 + depth * 17 }}>
        {group ? <button className="disclosure" type="button" aria-label={`${open ? 'Collapse' : 'Expand'} ${name}`} aria-expanded={open} onClick={() => toggle(node.id)}>
          <ChevronRight size={15} className={open ? 'is-open' : ''} />
        </button> : <span className="disclosure-spacer" />}
        <button className="node-button" type="button" title={name} aria-pressed={selected === node.id} onClick={() => onSelect(node.id)}>
          {group ? <Layers3 size={15} /> : <Box size={15} />}<span>{name}</span>
        </button>
      </div>
      {group && open && <ul>{node.children.map(child => renderNode(child, depth + 1))}</ul>}
    </li>;
  }

  return <section className="panel assembly-panel" aria-label="Assembly hierarchy">
    <h2>Assembly <span className="panel-meta">({index.parts} {index.parts === 1 ? 'part' : 'parts'})</span></h2>
    {roots.length > 0 && (onExportNames || onImportNames) && <div className="part-actions names-actions" role="group" aria-label="Part names CSV">
      {onExportNames && <button className="button" type="button" onClick={onExportNames}><Download size={14} /> Export names CSV</button>}
      {onImportNames && <>
        <button className="button" type="button" onClick={() => namesPicker.current?.click()}><Upload size={14} /> Import names CSV</button>
        <input ref={namesPicker} type="file" accept=".csv,text/csv" hidden onChange={event => {
          const file = event.currentTarget.files?.[0];
          event.currentTarget.value = '';
          if (file) onImportNames(file);
        }} />
      </>}
    </div>}
    {namesMessage && <p className={`name-note names-feedback${namesMessage.error ? ' is-error' : ''}`} role={namesMessage.error ? 'alert' : 'status'}>{namesMessage.text}</p>}
    {roots.length > 0 ? <ul className="assembly-tree">{roots.map(root => renderNode(root, 0))}</ul> : <p className="panel-empty">Import a STEP file to view its assembly.</p>}
  </section>;
}

const materials = [
  ['Violet', '#6651cf'],
  ['Silver', '#a7aeb9'],
  ['Graphite', '#41444d'],
  ['Stone', '#cec7bc'],
  ['Brass', '#caab54'],
];

type Inspection = { color: string | null; offset: readonly [number, number, number]; scale: number };

function DisplayNameEditor({ name, original, onSave }: { name: string; original: string; onSave: (value: string) => void }) {
  const id = useId();
  const [draft, setDraft] = useState<string | null>(null);
  const nextName = (draft ?? name).trim();
  const changed = draft !== null && (nextName || original) !== name;

  return <form className="name-editor" onSubmit={event => {
    event.preventDefault();
    if (changed) onSave(nextName === original ? '' : nextName);
    setDraft(null);
  }}>
    <label htmlFor={id}>Display name</label>
    <div className="name-editor-row">
      <input id={id} type="text" value={draft ?? name} autoComplete="off" aria-describedby={`${id}-note`}
        onChange={event => setDraft(event.target.value)} />
      <button className="button" type="submit" aria-label="Save display name" disabled={!changed}>Save</button>
    </div>
    {name !== original && <p className="name-note">Original STEP name <span>{original}</span></p>}
    <p id={`${id}-note`} className="name-note">Clear and save to use the STEP name.</p>
  </form>;
}

export function PartPanel({ part, details, canUndo, onColor, onRestore, onUndo, displayName, onDisplayName }: {
  part?: CadOccurrence;
  details?: Inspection | null;
  canUndo: boolean;
  onColor: (color: string) => void;
  onRestore: () => void;
  onUndo: () => void;
  displayName?: string;
  onDisplayName?: (value: string) => void;
}) {
  const changed = details && (details.offset.some(value => Math.abs(value) > 0.05) || Math.abs(details.scale - 1) > 0.005);
  const name = displayName ?? part?.name;
  return <section className="panel part-panel" aria-label="Selection properties">
    <h2>{part?.children.length ? 'Selected assembly' : 'Selected part'}</h2>
    {part ? <>
      <div className="part-name">{part.children.length ? <Layers3 size={19} /> : <Box size={19} />}<span title={name}>{name}</span></div>
      {onDisplayName && <DisplayNameEditor key={part.id} name={displayName ?? part.name} original={part.name} onSave={onDisplayName} />}
      {changed && <p className="transform-readout">Δ {details.offset.map(value => value.toFixed(1)).join(' / ')} mm <span>·</span> {details.scale.toFixed(2)}×</p>}
      <div className="material-label">Material</div>
      <div className="swatches" role="group" aria-label="Material color">
        {materials.map(([name, color]) => <button key={color} className="swatch" type="button" title={name} aria-label={name}
          aria-pressed={details?.color?.toLowerCase() === color} style={{ backgroundColor: color }} onClick={() => onColor(color)} />)}
      </div>
      <div className="part-actions">
        <button className="button" type="button" onClick={onRestore}><RotateCcw size={16} /> {part.children.length ? 'Restore assembly' : 'Restore part'}</button>
      </div>
    </> : <p className="panel-empty">Select a part from the assembly or tap it in the preview.</p>}
    {canUndo && <button className="button undo-button" type="button" onClick={onUndo}><Undo2 size={16} /> Undo last edit</button>}
  </section>;
}
