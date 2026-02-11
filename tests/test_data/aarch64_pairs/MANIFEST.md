# AArch64 Test Pairs

Each test case has:
- `.c` — C source
- `.s` — AArch64 assembly (clang -O0)
- `.ll` — Full LLVM IR (clang -O0 -emit-llvm)
- `_func.ll` — Extracted function IR (llvm-extract)

| Name | Description |
|------|-------------|
| `add_one` | Simple arithmetic: increment by 1 |
| `max_val` | Conditional: return maximum of two values |
| `sum_array` | Loop: sum array elements |
| `array_add_val` | Loop with pointer: add value to array elements (Forklift paper example) |
| `factorial` | Loop: iterative factorial |
| `fibonacci` | Loop with multiple variables: iterative fibonacci |
| `bubble_sort` | Nested loops: bubble sort |
| `abs_val` | Simple conditional: absolute value |
