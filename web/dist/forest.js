// Exact inference for the exported sklearn RandomForestClassifier.
export function predictForest(model, pixels) {
  if (!Array.isArray(pixels) || pixels.length !== model.features || pixels.some(v => !Number.isFinite(v) || v < 0 || v > 16)) {
    throw new Error('Cần đúng 64 giá trị pixel trong khoảng 0–16.');
  }
  const input = Float32Array.from(pixels);
  const probabilities = new Array(model.classes.length).fill(0);
  for (const tree of model.trees) {
    let node = 0;
    while (tree.left[node] !== -1) {
      node = input[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    }
    const value = tree.value[node];
    if (typeof value === 'number') probabilities[value] += 1;
    else value.forEach((p, i) => { probabilities[i] += p; });
  }
  probabilities.forEach((p, i) => { probabilities[i] = p / model.trees.length; });
  let winner = 0;
  probabilities.forEach((p, i) => { if (p > probabilities[winner]) winner = i; });
  return {label: model.classes[winner], probabilities};
}
