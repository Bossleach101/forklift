import re


def normalize_structs(llvm_ir):
    if not llvm_ir:
        return llvm_ir
    struct_dict = {}
    counter = 0
    normalized_ir = ""
    struct_pattern = re.compile(r"(%struct\.[a-zA-Z0-9_]+)")
    for line in llvm_ir.split("\n"):
        for match in struct_pattern.findall(line):
            if match not in struct_dict:
                struct_dict[match] = f"%struct.struct{counter}"
                counter += 1
        normalized_line = struct_pattern.sub(lambda m: struct_dict.get(m.group(1), m.group(1)), line)
        normalized_ir += normalized_line + "\n"
    return normalized_ir


def truncate_ir_output(ir_text: str) -> str:
    """Truncate generated LLVM IR after the function body closes.

    The model sometimes enters a degenerate repetitive loop after the
    closing ``}`` of the function definition, emitting endless
    ``declare`` lines or other garbage.  This utility keeps everything
    up to and including the *first* ``}`` that appears on a line by
    itself (the standard closing brace of a ``define`` block).

    Preamble lines before ``define`` (struct definitions, globals) are
    preserved.
    """
    if not ir_text:
        return ir_text

    lines = ir_text.split("\n")
    result: list[str] = []
    in_function = False

    for line in lines:
        stripped = line.strip()

        # Detect function start
        if stripped.startswith("define "):
            in_function = True

        result.append(line)

        # Detect function end: a lone "}" on its own line
        if in_function and stripped == "}":
            break

    return "\n".join(result)

class InferenceDataset:
    def __init__(self, data, compilers_keys=None):
        from .asm import AsmAdder, FuncDataclass
        self._FuncDataclass = FuncDataclass
        self.data = data
        self.asm_adder = AsmAdder(also_do_real=True, compilers_keys=compilers_keys)
        
    def __iter__(self):
        for instance in self.data:
            func_def = instance['func_def']
            deps = instance['deps']
            fname = instance['fname']
            e = self._FuncDataclass(func_def=func_def, fname=fname, func_head_types=func_def.split('{')[0], path='',
                              func_head=None, signature=None,
                              real_deps=deps)
            self.asm_adder.add_asm(e)
            e = e.dict()
            self._fix(e)
            yield e

    def _fix(self, row):
        row['asm'] = [{'target': target, 'code': code['func_asm'] if code else None} for (target, code) in
                      row['asm'].items()]

        # Flip dict due to format stored in HF
        new_asm = {'target': [], 'code': []}
        for e in row['asm']:
            new_asm['target'].append(e['target'])
            new_asm['code'].append(e['code'])
        row['asm'] = new_asm
        return row

    def __index__(self, idx):
        return self.data[idx]


