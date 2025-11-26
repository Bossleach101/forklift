import sh
from dataclasses import dataclass, asdict
from abc import ABC
from koda import Ok, Err, Result
import re
from typing import Dict, Optional, List
import operator
import itertools
from dataclasses import asdict, dataclass
import os
from lm_dataformat import Archive
from lm_dataformat import Reader
from koda import Ok, Err
from dataclasses import dataclass, fields

from typing import List, Optional, Dict

from copy import deepcopy
@dataclass
class AsmTarget:
    impl: str
    bits: int
    lang: str
    o: str

    def __post_init__(self):
        assert self.impl in ['gcc', 'clang']
        assert self.bits in [32, 64]
        assert self.lang in ['masm', 'gas', 'llvm']
        assert self.o in ['0', '1', '2', '3', 'fast', 'g', 'fast', 's', 'z']

    def dict(self):
        return asdict(self)


@dataclass
class FuncAsm:
    pre_asm: str  # asm directives before, and also e.g. global variable declarations needed to compile llvm functions
    func_asm: str  # asm of function itself
    post_asm: str  # asm directives after the function itself
    target: AsmTarget

    def dict(self):
        return asdict(self)


class Compiler:
    def __init__(self, arch, o, lang, bits=64, fPIC=False):
        self.arch = arch
        self.o = o
        self.bits = bits
        self.lang = lang
        self.fPIC = fPIC

    def get_func_asm(self, all_required_c_code, fname, output_path=None) -> Result[FuncAsm, BaseException]:
        return self._get_func_asm(all_required_c_code, fname, output_path, arch=self.arch, o=self.o, bits=self.bits)

    def _get_func_asm(self, all_required_c_code, fname, output_path, arch, o, bits) -> Result[FuncAsm, BaseException]:
        raise NotImplementedError

    def _asm_replace_constants_with_literals(self, all_asm, func_asm):
        raise NotImplementedError

    @classmethod
    def factory(cls, impl, *args, **kwargs):
        if impl == 'gcc':
            return GCC(*args, **kwargs)
        elif impl == 'clang':
            return Clang(*args, **kwargs)
        raise NotImplementedError(f'impl = {impl}')


class GASCompiler(ABC, Compiler):

    def get_comment_sym(self):
        if self.lang == 'gas':
            if self.arch == 'arm':
                return '@'
            return '#'
        elif self.lang == 'llvm':
            return ';'
        else:
            raise ValueError(f'lang = {self.lang}')

    def _asm_replace_constants_with_literals(self, all_asm, func_asm):
        all_asm = all_asm.decode("utf-8")
        asm_to_add = []
        for symbol in set(re.compile(r'\.LC[0-9]*').findall(func_asm)):  # TODO: move, compile once
            for e in re.findall(rf'\.{symbol.replace(".", "")}:[\r\n]+([^\r\n]+)', all_asm):
                asm_to_add.append(symbol + ': ' + e)
                break
        for symbol in set(re.compile(r'a\.[0-9]*').findall(func_asm)):  # TODO: move, compile once
            for e in re.findall(f'{symbol}:[\r\n]+([^\r\n]+)', all_asm):
                asm_to_add.append(symbol + ': ' + e)
                break
        return func_asm + '\n' + '\n'.join(asm_to_add) + '\n'

    def _gas_get_func_asm_from_all_asm(self, fname, all_asm):

        def strip_comments(code, comment_sym):  # only support simple commands, asm
            res = []
            for l in code.splitlines():
                without_comments = l.split(comment_sym)[0]
                if len(without_comments.split()) > 0:
                    res.append(without_comments)
            return '\n'.join(res)

        if self.arch == 'arm':
            func = [f'.global {fname}', f'.type {fname}, %function']
        else:
            func = [f'.globl {fname}', f'.type {fname}, @function']
        inside_func = False
        after_func = False
        pre_asm = []
        post_asm = []
        for l in all_asm.splitlines():
            if l.startswith(f'{fname}:'):
                inside_func = True
            if inside_func:
                func.append(l)
            elif after_func:
                post_asm.append(l)
            else:
                if self.arch == 'arm':
                    if f'.global {fname}' not in l:
                        pre_asm.append(l)
                else:
                    if f'.globl {fname}' not in l:
                        pre_asm.append(l)
            if inside_func and '.cfi_endproc' in l:
                inside_func = False
                after_func = True
            elif self.arch == 'riscv' and '.cfi_endproc' not in all_asm and ('jr\tra' in l or 'ret' in l or 'jr ra' in l):
                inside_func = False
                after_func = True

        pre_asm = '\n'.join(pre_asm) + '\n'
        func_asm = '\n'.join(func) + '\n'
        comment_sym = self.get_comment_sym()
        func_asm = strip_comments(func_asm, comment_sym=comment_sym)
        post_asm = '\n'.join(post_asm) + '\n'

        return pre_asm, func_asm, post_asm


class GCC(GASCompiler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, lang='gas', **kwargs)
        self.x86_64 = sh.gcc
        try:
            self.arm_64 = sh.aarch64_linux_gnu_gcc  # sudo apt install gcc-aarch64-linux-gnu
            self.riscv_64 = sh.riscv64_linux_gnu_gcc # sudo apt install gcc-riscv64-linux-gnu
            self.arm_32 = sh.arm_linux_gnueabi_gcc # sudo apt install  gcc-arm-linux-gnueabi
        except:
            pass
    def _get_func_asm(self, all_required_c_code, fname, output_path, arch, o, bits) -> Result[FuncAsm, BaseException]:
        lang = 'gas'
        if arch == 'arm' and bits == 64:
            backend = self.arm_64
        elif arch == 'x86' and bits == 64:
            backend = self.x86_64
        elif arch == 'riscv' and bits == 64:
            backend = self.riscv_64
        elif arch == 'arm' and bits == 32:
            backend = self.arm_32
        else:
            raise NotImplementedError(f'arch = {arch}, bits = {bits}')
        try:
            extra_cmd = ['-fPIC'] if self.fPIC else []
            out = backend('-S', *extra_cmd, f'-O{o}', '-x', 'c', '-o', '/dev/stdout', '-', _in=all_required_c_code)
        except BaseException as e:
            return Err(e)

        if self.fPIC:
            out = out.stdout.decode()

            class _anonclass:
                def __init__(self):
                    self.func_asm = out
                    return
                def dict(self):
                    return {'func_asm': self.func_asm}
            return Ok(_anonclass())
        pre_asm, func_asm, post_asm = self._gas_get_func_asm_from_all_asm(all_asm=out, fname=fname)

        if not (arch == 'arm' and bits == 32):
            func_asm = self._asm_replace_constants_with_literals(all_asm=out.stdout, func_asm=func_asm)
        if self.fPIC:
            for l in post_asm.splitlines():
                if '.comm' in l:
                    func_asm += l + '\n'
        func_asm = FuncAsm(pre_asm=pre_asm, func_asm=func_asm, post_asm=post_asm, target=AsmTarget(impl='gcc',
                                                                                                   bits=bits,
                                                                                                   lang=lang,
                                                                                                   o=o))
        return Ok(func_asm)


class Clang(GASCompiler):
    def __init__(self, *args, emit_llvm=False, **kwargs):
        lang = 'llvm' if emit_llvm else 'gas'
        super().__init__(*args, lang=lang, **kwargs)
        self.clang = sh.clang  # sudo apt install clang
        self.emit_llvm = emit_llvm
        self.emit_llvm_flag = '-emit-llvm' if emit_llvm else ''

    def get_comment_sym(self):
        if self.lang == 'gas':
            if self.arch == 'arm':
                return '//'
            return '#'
        elif self.lang == 'llvm':
            return ';'
        else:
            raise ValueError(f'lang = {self.lang}')

    def _get_all_llvm_ir(self, all_required_c_code):
        assert self.emit_llvm
        arch = self.arch
        o = self.o
        bits = self.bits
        extra_cmd = []
        all_required_c_code = all_required_c_code.replace('static ', ' ').replace('static\n', '\n').replace('static\t',
                                                                                                            '\t')
        # clang doesn't return assembly in some cases
        if arch == 'x86' and bits == 64:
            backend = self.clang
        elif arch == 'arm' and bits == 64:
            backend = self.clang
            extra_cmd.append('--target=aarch64')
        elif arch == 'riscv' and bits == 64:
            backend = self.clang
            extra_cmd.append('--target=riscv64')
        elif arch == 'arm' and bits == 32:
            backend = self.clang
            extra_cmd.append('--target=arm-linux-gnueabi')
            # extra_cmd.append('--target=arm-linux-gnueabihf')
        else:
            raise NotImplementedError(f'arch = {arch}, bits = {bits}')
        try:
            if self.fPIC:
                extra_cmd.append('-fPIC')
            out = backend(*extra_cmd, '-S', self.emit_llvm_flag, f'-O{o}', '-x', 'c', '-o', '/dev/stdout', '-',
                          _in=all_required_c_code)
            if self.fPIC:
                out = out.stdout.decode()

                class _anonclass:
                    def __init__(self):
                        self.func_asm = out
                        return

                    def dict(self):
                        return {'func_asm': self.func_asm}

                return Ok(_anonclass())
        except BaseException as e:
            return Err(e)
        return Ok(out)


    def _get_func_asm(self, all_required_c_code, fname, output_path, arch, o, bits) -> Result[FuncAsm, BaseException]:
        extra_cmd = []
        all_required_c_code = all_required_c_code.replace('static ', ' ').replace('static\n', '\n').replace('static\t', '\t')
        # clang doesn't return assembly in some cases
        if arch == 'x86' and bits == 64:
            backend = self.clang
        elif arch == 'arm' and bits == 64:
            backend = self.clang
            extra_cmd.append( '--target=aarch64')
        elif arch == 'riscv' and bits == 64:
            backend = self.clang
            extra_cmd.append( '--target=riscv64')
        elif arch == 'arm' and bits == 32:
            backend = self.clang
            extra_cmd.append('--target=arm-linux-gnueabi')
        else:
            raise NotImplementedError(f'arch = {arch}, bits = {bits}')
        try:
            if self.fPIC:
                extra_cmd.append('-fPIC')
            out = backend(*extra_cmd, '-S', self.emit_llvm_flag, f'-O{o}', '-x', 'c', '-o', '/dev/stdout', '-',
                          _in=all_required_c_code)
            if self.fPIC:
                out = out.stdout.decode()
                class _anonclass:
                    def __init__(self):
                        self.func_asm = out
                        return
                    def dict(self):
                        return {'func_asm': self.func_asm}

                return Ok(_anonclass())
        except BaseException as e:
            return Err(e)
        try:
            if self.emit_llvm:
                try:
                    func_asm = self._llvm_get_func_asm_from_all_asm_using_llvm_extract(all_asm=out, fname=fname)
                except BaseException as e:
                    return Err(e)
                pre_asm = ''
                post_asm = ''
            else:
                pre_asm, func_asm, post_asm = self._gas_get_func_asm_from_all_asm(all_asm=out, fname=fname)
                before, func_end, after = func_asm.partition('.Lfunc_end0:\n')
                new_func_asm = before
                if '.cfi_endproc' in func_asm and '.cfi_endproc' not in new_func_asm:
                    new_func_asm += '\t.cfi_endproc\n'
                    after = '\n'.join([line for line in after.splitlines() if '.cfi_endproc' not in line])
                func_asm = new_func_asm
                post_asm = func_end + after + '\n' + post_asm
        except RuntimeError as e:
            return Err(e)
        if not self.emit_llvm:
            if not (arch == 'arm' and bits == 32):
                if isinstance(out, str):
                    func_asm = self._asm_replace_constants_with_literals(all_asm=out, func_asm=func_asm)
                else:
                    func_asm = self._asm_replace_constants_with_literals(all_asm=out.stdout, func_asm=func_asm)
        func_asm = FuncAsm(pre_asm=pre_asm, func_asm=func_asm, post_asm=post_asm, target=AsmTarget(impl='clang',
                                                                                                   bits=bits,
                                                                                                   lang='llvm' if self.emit_llvm else 'gas',
                                                                                                   o=o))
        return Ok(func_asm)

    def _asm_replace_constants_with_literals(self, all_asm, func_asm):
        if not isinstance(all_asm, str):
            all_asm = all_asm.decode("utf-8")
        symbols = set(re.compile(r'\.LC[0-9A-Z_]*').findall(func_asm)) | set(re.compile(r'\.L\.[a-z0-9]*').findall(func_asm))
        asm_to_add = []
        for symbol in symbols:
            for e in re.findall(f'{symbol}:[\r\n]+([^\r\n]+)', all_asm):
                asm_to_add.append(symbol + ':')
                e = re.sub(fr'{self.get_comment_sym()}.*$', '', e, flags=re.MULTILINE)
                asm_to_add.append(e)
                break
        if not func_asm.endswith('\n'):
            func_asm = func_asm + '\n'
        return func_asm + '\n'.join(asm_to_add) + '\n'

    @staticmethod
    def normalize_structs(llvm_ir):
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

    @classmethod
    def _llvm_get_func_asm_from_all_asm_using_llvm_extract(cls, fname, all_asm):
        llvm_extract = sh.llvm_extract
        out = llvm_extract('-S', f'--func={fname}', _in=all_asm)

        ir = out.stdout.decode() if isinstance(out.stdout, bytes) else out.stdout
        filtered_ir = []
        for line in ir.splitlines():
            if len(line.splitlines()) == 0:
                continue
            if line.startswith(';'):
                continue
            if line.startswith('source_filename') or line.startswith('target datalayout') or line.startswith('target triple'):
                continue
            if line.startswith('attributes'):
                continue
            if line.startswith('!'):
                continue
            filtered_ir.append(line)
        filtered_ir = '\n'.join(filtered_ir)
        ir = re.sub(r'#\S+', '', filtered_ir)
        pattern = r"(,?\s*![a-zA-Z0-9]+)"
        ir = re.sub(pattern, "", ir)
        return ir


@dataclass
class IOPair:
    input: Dict
    output: Dict
    dummy_funcs: Optional[str] = None
    dummy_funcs_seed: Optional[int] = None

    def dict(self):
        return asdict(self)

    @classmethod
    def group_by_seed(cls, iopairs: List['IOPair']):
        get_attr = operator.attrgetter('dummy_funcs_seed')
        grouped_by = [list(g) for k, g in itertools.groupby(sorted(iopairs, key=get_attr), get_attr)]
        return grouped_by

@dataclass
class FuncDataclass:
    path: str
    func_def: str
    func_head: str
    fname: str
    signature: List[str]
    doc: Optional[str] = None
    angha_error: Optional[str] = None
    real_error: Optional[str] = None
    asm: Optional[Dict[str, Optional[FuncAsm]]] = None
    angha_deps: Optional[str] = None
    real_deps: Optional[str] = None
    angha_io_pairs: List[IOPair] = None
    real_io_pairs: List[IOPair] = None
    angha_io_error: Optional[str] = None
    real_io_error: Optional[str] = None
    angha_exe_wrapper: Optional[str] = None
    real_exe_wrapper: Optional[str] = None
    angha_io_pairs_are_trivial: bool = False
    real_io_pairs_are_trivial: bool = False
    angha_iospec: Optional[Dict] = None
    real_iospec: Optional[Dict] = None
    ref: str = 'master'
    func_head_types: str = ''

    def dict(self):
        return asdict(self)

    def get_fname_tmp_fix(self):
        return self.func_head.split('(')[-2].split()[-1].replace('*', '')

    @staticmethod
    def dict_get_fname_tmp_fix(row_dict: Dict):
        return row_dict['func_head'].split('(')[-2].split()[-1].replace('*', '')

    @classmethod
    def legacy_init(cls, **kwargs):
        del kwargs['io_pairs']
        return cls(**kwargs)


class AsmAdder:
    def __init__(self, compilers=None, also_do_real=False, replace_asm=False, compilers_keys=None):
        self.compilers = self.setup_compilers() if not compilers else compilers
        if compilers_keys:
            filtered_compilers = {}
            for k in self.compilers:
                if k in compilers_keys:
                    filtered_compilers[k] = self.compilers[k]
            self.compilers = filtered_compilers
        self.also_do_real = also_do_real
        self.replace_asm = replace_asm

    def add_asm_to_dict(self, fd_row: Dict):
        asm_to_add = {}
        n_ok = {}
        for compiler in self.compilers:
            if fd_row['synth_deps'] is not None and f'angha_{compiler}' not in fd_row['asm']:
                all_required_c_code =fd_row['synth_deps'] + '\n' + fd_row['func_def']
                all_required_c_code = all_required_c_code.replace('inline', ' ')
                fname = fd_row['fname'] if ('func_head' not in fd_row or not fd_row[
                    'func_head']) else FuncDataclass.dict_get_fname_tmp_fix(fd_row)
                res_asm_angha = self.compilers[compiler].get_func_asm(all_required_c_code=all_required_c_code,
                                                                      fname=fname)
                if isinstance(res_asm_angha, Ok):
                    asm_to_add[f'angha_{compiler}'] = res_asm_angha.val
                    n_ok[f'angha_{compiler}'] = 1
                else:
                    asm_to_add[f'angha_{compiler}'] = None
                    n_ok[f'angha_{compiler}'] = 0

            if fd_row['real_deps'] is not None and self.also_do_real and f'real_{compiler}' not in fd_row['asm']:
                all_required_c_code =fd_row['real_deps'] + '\n' + fd_row['func_def']
                all_required_c_code = all_required_c_code.replace('inline', ' ')
                fname = fd_row['fname'] if ('func_head' not in fd_row or not fd_row['func_head']) else FuncDataclass.dict_get_fname_tmp_fix(fd_row)
                res_asm_real = self.compilers[compiler].get_func_asm(all_required_c_code=all_required_c_code,
                                                                     fname=fname)

                if isinstance(res_asm_real, Ok):
                    asm_to_add[f'real_{compiler}'] = res_asm_real.val
                    n_ok[f'real_{compiler}'] = 1
                else:
                    asm_to_add[f'real_{compiler}'] = None
                    n_ok[f'real_{compiler}'] = 0
        return asm_to_add

    def add_asm(self, fd_dataclass: FuncDataclass):
        asm_to_add = {}
        n_ok = {}
        for compiler in self.compilers:
            if fd_dataclass.angha_deps is not None and (not fd_dataclass.asm or not fd_dataclass.asm[f'angha_{compiler}']):
                all_required_c_code = fd_dataclass.angha_deps + '\n' + fd_dataclass.func_def
                all_required_c_code = all_required_c_code.replace('inline', ' ')
                res_asm_angha = self.compilers[compiler].get_func_asm(all_required_c_code=all_required_c_code,
                                                                      fname=fd_dataclass.get_fname_tmp_fix())
                if isinstance(res_asm_angha, Ok):
                    asm_to_add[f'angha_{compiler}'] = res_asm_angha.val
                    n_ok[f'angha_{compiler}'] = 1
                else:
                    asm_to_add[f'angha_{compiler}'] = None
                    n_ok[f'angha_{compiler}'] = 0

            if fd_dataclass.real_deps is not None and self.also_do_real and (not fd_dataclass.asm or not fd_dataclass.asm[f'real_{compiler}']):
                all_required_c_code = fd_dataclass.real_deps + '\n' + fd_dataclass.func_def
                all_required_c_code = all_required_c_code.replace('inline', ' ')
                fname = fd_dataclass.fname if not fd_dataclass.func_head else fd_dataclass.get_fname_tmp_fix()
                res_asm_real = self.compilers[compiler].get_func_asm(all_required_c_code=all_required_c_code,
                                                                     fname=fname)

                if isinstance(res_asm_real, Ok):
                    asm_to_add[f'real_{compiler}'] = res_asm_real.val
                    n_ok[f'real_{compiler}'] = 1
                else:
                    asm_to_add[f'real_{compiler}'] = None
                    n_ok[f'real_{compiler}'] = 0

        if self.replace_asm:
            fd_dataclass.asm = asm_to_add
        else:
            if not fd_dataclass.asm:
                fd_dataclass.asm = {}
            for k in asm_to_add:
                if asm_to_add[k]:
                    asm_to_add[k] = asm_to_add[k].dict()
                fd_dataclass.asm[k] = asm_to_add[k]

    @staticmethod
    def setup_compilers():  # adding riscv, clang IR O3
        gcc_x86_O0 = Compiler.factory('gcc', arch='x86', o='0')
        gcc_x86_O3 = Compiler.factory('gcc', arch='x86', o='3')
        gcc_x86_Os = Compiler.factory('gcc', arch='x86', o='s')
        gcc_arm_O0 = Compiler.factory('gcc', arch='arm', o='0')
        gcc_arm_Os = Compiler.factory('gcc', arch='arm', o='s')
        gcc_arm_O3 = Compiler.factory('gcc', arch='arm', o='3')
        clang_x86_O0 = Compiler.factory('clang', arch='x86', o='0')
        clang_x86_O3 = Compiler.factory('clang', arch='x86', o='3')
        clang_ir_O0 = Compiler.factory('clang', arch='x86', o='0', emit_llvm=True)
        clang_ir_Oz = Compiler.factory('clang', arch='x86', o='z', emit_llvm=True)
        clang_arm_O0 = Compiler.factory('clang', arch='arm', o='0')
        clang_arm_O3 = Compiler.factory('clang', arch='arm', o='3')
        gcc_riscv_O0 = Compiler.factory('gcc', arch='riscv', o='0')
        gcc_riscv_O3 = Compiler.factory('gcc', arch='riscv', o='3')
        clang_riscv_O0 = Compiler.factory('clang', arch='riscv', o='0')
        clang_riscv_O3 = Compiler.factory('clang', arch='riscv', o='3')
        clang_ir_O3 = Compiler.factory('clang', arch='x86', o='3', emit_llvm=True)

        gcc_arm32_O0 = Compiler.factory('gcc', arch='arm', o='0', bits=32)
        gcc_arm32_O3 = Compiler.factory('gcc', arch='arm', o='3', bits=32)
        clang_arm32_O0 = Compiler.factory('clang', arch='arm', o='0', bits=32)
        clang_arm32_O3 = Compiler.factory('clang', arch='arm', o='3', bits=32)
        compilers = {'gcc_x86_O0': gcc_x86_O0, 'gcc_x86_O3': gcc_x86_O3,
                     'gcc_x86_Os': gcc_x86_Os,
                     'gcc_arm_O0': gcc_arm_O0,
                     'gcc_arm_Os': gcc_arm_Os,
                     'gcc_arm_O3': gcc_arm_O3,
                     'clang_x86_O0': clang_x86_O0, 'clang_x86_O3': clang_x86_O3, 'clang_ir_O0': clang_ir_O0,
                     'clang_ir_Oz': clang_ir_Oz,
                     'clang_arm_O0': clang_arm_O0, 'clang_arm_O3': clang_arm_O3,

                     'gcc_riscv_O0': gcc_riscv_O0,

                     'gcc_riscv_O3': gcc_riscv_O3,
                     'clang_riscv_O0': clang_riscv_O0,
                     'clang_riscv_O3': clang_riscv_O3,
                     'clang_ir_O3': clang_ir_O3,
                     'gcc_arm32_O0': gcc_arm32_O0,
                     'gcc_arm32_O3': gcc_arm32_O3,
                     'clang_arm32_O0': clang_arm32_O0,
                     'clang_arm32_O3': clang_arm32_O3
                     }


        fpic_compilers = {}
        for k in compilers:
            fpic = deepcopy(compilers[k])
            fpic.fPIC = True
            fpic_compilers[f'{k}_fPIC'] = fpic
        for k in fpic_compilers:
            compilers[k] = fpic_compilers[k]
        return compilers
