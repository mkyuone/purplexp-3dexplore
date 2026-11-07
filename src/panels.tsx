import { Box, ChevronRight, Layers3, RotateCcw, Undo2 } from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';
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

export function AssemblyPanel({ roots, index, selected, onSelect }: {
  roots: CadOccurrence[];
  index: AssemblyIndex;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(() => new Set(roots.map(root => root.id)));

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
    return <li key={node.id}>
      <div className={`assembly-row${selected === node.id ? ' is-selected' : ''}`} style={{ paddingInlineStart: 4 + depth * 17 }}>
        {group ? <button className="disclosure" type="button" aria-label={`${open ? 'Collapse' : 'Expand'} ${node.name}`} aria-expanded={open} onClick={() => toggle(node.id)}>
          <ChevronRight size={15} className={open ? 'is-open' : ''} />
        </button> : <span className="disclosure-spacer" />}
        <button className="node-button" type="button" title={node.name} aria-pressed={selected === node.id} onClick={() => onSelect(node.id)}>
          {group ? <Layers3 size={15} /> : <Box size={15} />}<span>{node.name}</span>
        </button>
      </div>
      {group && open && <ul>{node.children.map(child => renderNode(child, depth + 1))}</ul>}
    </li>;
  }

  return <section className="panel assembly-panel" aria-label="Assembly hierarchy">
    <h2>Assembly <span className="panel-meta">({index.parts} {index.parts === 1 ? 'part' : 'parts'})</span></h2>
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

export function PartPanel({ part, details, canUndo, onColor, onRestore, onUndo }: {
  part?: CadOccurrence;
  details?: Inspection | null;
  canUndo: boolean;
  onColor: (color: string) => void;
  onRestore: () => void;
  onUndo: () => void;
}) {
  const changed = details && (details.offset.some(value => Math.abs(value) > 0.05) || Math.abs(details.scale - 1) > 0.005);
  return <section className="panel part-panel" aria-label="Selection properties">
    <h2>{part?.children.length ? 'Selected assembly' : 'Selected part'}</h2>
    {part ? <>
      <div className="part-name">{part.children.length ? <Layers3 size={19} /> : <Box size={19} />}<span title={part.name}>{part.name}</span></div>
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
