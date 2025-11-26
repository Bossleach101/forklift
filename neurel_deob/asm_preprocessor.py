"""
Plan

Clean up x86 assembly:   
    You can now use this approach to lift any x86-64 assembly to LLVM IR. Just make sure:

    Remove .cfi_* directives
    Use clean GAS (GNU Assembler) syntax
    Keep the .globl and .type declarations

"""
import re


class AsmPreprocessor:
    def __init__(self, asm_code: str = None, 
                target_function: str = None,
                file_path: str = None):
        
        if target_function is None:
            print("Warning: target_function is None, defaulting to 'f'")
            target_function = 'f'

        if asm_code:
            self.func_asm = self._get_target_function_asm(asm_code, target_function)
        elif file_path:
            with open(file_path, 'r') as f:
                asm_code = f.read()
            self.func_asm = self._get_target_function_asm(asm_code, target_function)
        else:
            raise ValueError("Either asm_code or file_path must be provided.")
    
    # Extracts assembly function from full asm code only
    def _get_target_function_asm(self, asm_code: str, target_function: str) -> str:
        
        lines = asm_code.split('\n')
        in_function = False
        raw_func_lines = []
        for line in lines:
            if re.match(rf'\.globl\s+{target_function}', line.strip()):
                in_function = True
                raw_func_lines.append(line)
            elif in_function and not re.match(rf'\.size\s+{target_function}\s*,\s*\.\s*-\s*{target_function}', line.strip()):
                raw_func_lines.append(line)
            else:
                in_function = False
        
        raw_func_lines= raw_func_lines[:-1]  # Remove the .LFE1: 
        raw_func_lines = [line for line in raw_func_lines if not self._to_be_removed(line)]
        return '\n'.join(raw_func_lines)
    
    # Remove Call Frame Information as not in training data (meta data)
    def _to_be_removed(self, line: str) -> bool:
        patterns = [
            r'^\s*\.cfi_.*$', # Matches .cfi directives
            r'^\s*\#.*$'# Matches comments
        ]
        for pattern in patterns:
            if re.match(pattern, line):
                return True
        return False
    

if __name__ == '__main__':
    p = AsmPreprocessor(file_path='../output1.s')
    print(p.func_asm)
