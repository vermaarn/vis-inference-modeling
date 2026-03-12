/**
 * Generate public/visualization-pngs.json from the visualizations folder.
 * Run from repo root: node explorer/scripts/generate-png-list.js
 * Or from explorer: node scripts/generate-png-list.js (adjust VIS_DIR if needed)
 */
const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '../..');
const visDir = path.join(repoRoot, 'data/images');
const outPath = path.join(repoRoot, 'explorer', 'public', 'visualization-pngs.json');

const names = fs.readdirSync(visDir)
  .filter((n) => n.endsWith('.png'))
  .sort((a, b) => {
    const aNum = parseInt(a.replace(/\D/g, '') || '0', 10);
    const bNum = parseInt(b.replace(/\D/g, '') || '0', 10);
    if (aNum !== bNum) return aNum - bNum;
    return a.localeCompare(b);
  });

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(names, null, 2));
console.log('Wrote', outPath, '(', names.length, 'PNGs)');
