// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import { pathToFileURL } from 'url';

const jsPath = process.argv[2];
if (!jsPath) {
  console.error('Usage: node test_velox_wasm.mjs <path-to-velox_wasm.js>');
  process.exit(1);
}

const absPath = jsPath.startsWith('/') ? jsPath : process.cwd() + '/' + jsPath;
const initModule = (await import(pathToFileURL(absPath).href)).default;
const Module = await initModule();
const engine = new Module.VeloxWasmEngine();

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`PASS: ${name}`);
  } catch (e) {
    failed++;
    console.error(`FAIL: ${name}: ${e.message}`);
  }
}

function assertOk(result) {
  const parsed = JSON.parse(result);
  if (!parsed.ok) throw new Error(parsed.error || 'test returned ok=false');
  return parsed;
}

test('expression evaluation', () => assertOk(engine.testExpressionEval()));
test('plan execution', () => assertOk(engine.testPlanExecution()));
test('aggregation', () => assertOk(engine.testAggregation()));
test('group by', () => assertOk(engine.testGroupBy()));
test('order by', () => assertOk(engine.testOrderBy()));
test('hash join', () => assertOk(engine.testHashJoin()));
test('timezone fallback', () => assertOk(engine.testTimezoneFallback()));
test('serialization', () => assertOk(engine.testSerialization()));

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
