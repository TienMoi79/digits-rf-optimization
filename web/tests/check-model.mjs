import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {predictForest} from '../dist/forest.js';
const load = async path => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const model = await load('../dist/data/forest.json');
const fixtures = await load('./model-fixtures.json');
const data = await load('../dist/data/dashboard.json');
let maxDelta = 0;
fixtures.inputs.forEach((pixels, index) => {
  const result = predictForest(model, pixels);
  assert.equal(result.label, fixtures.predictions[index]);
  result.probabilities.forEach((p, i) => {
    const delta = Math.abs(p - fixtures.probabilities[index][i]);
    maxDelta = Math.max(maxDelta, delta);
    assert.ok(delta < 1e-12);
  });
});
for (const invalid of [[], new Array(64).fill(NaN), new Array(64).fill(17), new Array(64).fill(-1)]) {
  assert.throws(() => predictForest(model, invalid));
}
assert.equal(model.trees.length, 100);
assert.equal(data.trials.length, 181);
assert.equal(data.metrics.find(r => r.label === 'selected').test_correct, 350);
assert.equal(fixtures.test_evaluations, 0);
console.log(`PASS: ${fixtures.inputs.length} JS/Python inference fixtures match; max probability delta ${maxDelta}; invalid pixels rejected.`);
