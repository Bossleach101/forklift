import re
from .asm import AsmAdder, FuncDataclass

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

class InferenceDataset:
    def __init__(self, data, compilers_keys=None):
        self.data = data
        self.asm_adder = AsmAdder(also_do_real=True, compilers_keys=compilers_keys)
        
    def __iter__(self):
        for instance in self.data:
            func_def = instance['func_def']
            deps = instance['deps']
            fname = instance['fname']
            e = FuncDataclass(func_def=func_def, fname=fname, func_head_types=func_def.split('{')[0], path='',
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


