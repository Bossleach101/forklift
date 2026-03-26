import re
from tokenizers import Tokenizer
import re
from .utils import normalize_structs

class DP:

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def source_goes_left(self, pair, legacy_opt3_ir=False):
        res = True
        if pair in ['opt3_c-c', 'arm_s-s', 's_c-c', 'arm_c-c', 'arm_opt3_c-c', 'riscv_c-c', 'riscv_opt3_c-c', 's_ir-ir',
                    'arm_ir-ir', 'riscv_ir-ir', 'riscv_opt3_ir_opt3-ir_opt3', 'arm_opt3_ir_opt3-ir_opt3', 'opt3_ir_opt3-ir_opt3',
                    'arm_opt3_ir_optz-ir_optz', 'riscv_opt3_ir_optz-ir_optz']:
            res = False
        if not legacy_opt3_ir and pair in ['opt3_ir_optz-ir_optz', 'opt3_ir-ir']:
            res = False
        return res

    def lang_special_token(self, lang, opt3_legacy=False):
        modifiers = []
        if 'clang_' in lang:
            lang = lang.replace('clang_', '')
            modifiers.append('clang')
        if lang == 's':
            token = 'intel'
        elif lang == 'opt':
            token = 'intel'
            modifiers.append('opt')
        elif lang == 'opt3':
            token = 'intel'
            modifiers.append('opt3')
        elif lang == 'c':
            token = 'c'
        elif 'arm' in lang:
            token = 'arm'
        elif 'riscv' in lang:
            token = 'riscv'
        elif 'ir' in lang:
            token = 'ir'
        else:
            raise ValueError(lang)
        if lang not in ['s', 'opt', 'opt3']:
            if 'opts' in lang:
                modifiers.append('opts')
            elif 'opt3' in lang:
                modifiers.append('opt3')
            elif 'oz' in lang:
                modifiers.append('oz')

        def _get_start_end(tok):
            _start = f'<{tok}>'
            _end = f'</{tok}>'
            if opt3_legacy and tok == 'opt3': # legacy bug
                _end = '</opt>'
            return _start, _end

        start, end = _get_start_end(token)
        modifiers = list(map(_get_start_end, modifiers))

        return start, end, modifiers

    @staticmethod
    def get_lang_from_pair(p, source_or_target):
        assert source_or_target in ['source', 'target']
        if source_or_target == 'source':
            s = p.split('-')[0]
            t = p.split('-')[1]
            assert t in s
            if s.count(t) > 1:  # clang vs c confusion
                s = s[::-1].replace(t[::-1], '', 1)[::-1]
                l = s[:-1]

            else:
                l = p.split('-')[0].replace(p.split('-')[1], '')[:-1]
        else:
            l = p.split('-')[1]
        return l

    def tokenize(self, source, target, pair, ids=True, no_tokenize=False, opt3_legacy=False):
        # one_sample must be a function definition (and ONLY a function definition)
        assert pair.count('-') < 2
        if '-' not in pair:
            raise RuntimeError('mono tokenization not supported')
        # source
        lang_source = self.get_lang_from_pair(pair, 'source')
        start_lang_tok, end_lang_tok, modifiers = self.lang_special_token(lang_source, opt3_legacy=opt3_legacy)
        if modifiers:
            modifiers_start = ' '.join([start for start, end in modifiers])
            modifiers_end = ' '.join(reversed([start for start, end in modifiers]))
            one_sample = f"{start_lang_tok} {modifiers_start} {source} {modifiers_end} {end_lang_tok}"
        else:
            one_sample = f"{start_lang_tok} {source} {end_lang_tok}"
        # target
        start_lang_tok, end_lang_tok, modifiers = self.lang_special_token(self.get_lang_from_pair(pair, 'target'), opt3_legacy=opt3_legacy)
        if modifiers:
            modifiers_start = ' '.join([start for start, end in modifiers])
            modifiers_end = ' '.join(reversed([start for start, end in modifiers]))
            one_sample_ref = f"{start_lang_tok} {modifiers_start} {target} {modifiers_end} {end_lang_tok}"
            if no_tokenize:
                return one_sample, one_sample_ref
            one_sample_ref_norm = self.tokenizer.normalizer.normalize_str(one_sample_ref)
            one_sample_ref_norm = one_sample_ref_norm.replace(f"{start_lang_tok} {modifiers_start} ", '').replace(f" {modifiers_end} {end_lang_tok}", '')
            one_sample_masked = f"{start_lang_tok} {modifiers_start} <mask:0> {modifiers_end} {end_lang_tok}"

        else:
            one_sample_ref = f"{start_lang_tok} {target} {end_lang_tok}"
            if no_tokenize:
                return one_sample, one_sample_ref
            one_sample_ref_norm = self.tokenizer.normalizer.normalize_str(one_sample_ref)
            one_sample_ref_norm = one_sample_ref_norm.replace(f"{start_lang_tok} ", '').replace(
                f" {end_lang_tok}", '')
            one_sample_masked = f"{start_lang_tok} <mask:0> {end_lang_tok}"

        if self.source_goes_left(pair):
            one_sample = f'{one_sample} {one_sample_masked}'
        else:
            one_sample = f'{one_sample_masked} {one_sample}'

        source_tokenized = self.tokenizer.encode(self.tokenizer.normalizer.normalize_str(one_sample))
        target_tokenized = self.tokenizer.encode(one_sample_ref_norm)
        if ids:
            source_tokenized = source_tokenized.ids
            target_tokenized = target_tokenized.ids
        else:
            source_tokenized = source_tokenized.tokens
            target_tokenized = target_tokenized.tokens
        return source_tokenized, target_tokenized


    def prepare(self, row, pair, asm_key='angha', return_target_length=False):
        if not return_target_length:
            one_sample = self.row_to_sample(row, pair, asm_key)
            return one_sample
        one_sample, t_length = self.row_to_sample(row, pair, asm_key, return_target_length=True)
        return one_sample, t_length

    def row_to_sample(self, row, pair, asm_key, return_target_length=False):
        source, target, tokenized_source, tokenized_target = self.get_par_data(row, pair, asm_key=asm_key, fPIC=False,
                                                                               tokenize_ids=True,
                                                                               do_normalize_ir_structs=True)
        if not return_target_length:
            return tokenized_source
        return tokenized_source, len(tokenized_target)

    def detokenize(self, one_sample, remove_mask=True):
        # We can't directly use decode(), need to remove some special tokens by hand (they can't be skipped as the others)
        detok = self.tokenizer.decode(one_sample, skip_special_tokens=False)
        detok = detok.replace('<eol> ', '\n').replace('<eol>', '\n').replace('<tab> ', '\t').replace('<tab>', '\t')
        if remove_mask:
            detok = detok.replace(
            '<mask:0>', '')
        detok = detok.replace('<pad>', '').replace('<s>', '').replace('</s>', '')
        detok = re.sub('# (/\w+)*', '', detok)
        detok = detok.replace('0x ', '0x').replace(' #', '').replace('return', 'return ').replace('return  ', 'return ')
        detok = detok.replace('static', '').replace('inline', '')
        detok = re.sub('# (/\w+)*', '', detok)
        detok = detok.replace('__attribute__((used))', '')
        return detok

    def get_asm_key(self, key, asm_key='angha', compiler='gcc', fPIC=False):
        assert compiler in ['gcc', 'clang']
        if key in ['x86', 's']:
            full_key = f'{asm_key}_{compiler}_x86_O0'
        elif key == 'opt':
            full_key = f'{asm_key}_{compiler}_x86_Os'
        elif key == 'arm':
            full_key  = f'{asm_key}_{compiler}_arm_O0'
        elif key == 'opt3':
            full_key = f'{asm_key}_{compiler}_x86_O3'
        elif key == 'arm_opts':
            full_key = f'{asm_key}_{compiler}_arm_Os'
        elif key == 'arm_opt3':
            full_key = f'{asm_key}_{compiler}_arm_O3'
        elif key == 'riscv':
            full_key = f'{asm_key}_{compiler}_riscv_O0'
        elif key == 'riscv_opts':
            full_key = f'{asm_key}_{compiler}_riscv_Os'
        elif key == 'riscv_opt3':
            full_key = f'{asm_key}_{compiler}_riscv_O3'
        elif key == 'ir':
            full_key = f'{asm_key}_{compiler}_ir_O0'
        elif key == 'ir_opts':
            full_key = f'{asm_key}_{compiler}_ir_Os'
        elif key == 'ir_optz':
            full_key = f'{asm_key}_{compiler}_ir_Oz'
        elif key == 'ir_opt3':
            full_key = f'{asm_key}_{compiler}_ir_O3'
        elif key == 'arm32':
            full_key = f'{asm_key}_{compiler}_arm32_O0'
        elif key == 'arm32_opt3':
            full_key = f'{asm_key}_{compiler}_arm32_O3'
        else:
            raise RuntimeError(key)
        if fPIC:
            full_key += '_fPIC'
        return full_key

    def get_asm(self, key, row=None, asm_key='angha', compiler='gcc', fPIC=False, do_normalize_ir_structs=False):
        assert compiler in ['gcc', 'clang']
        if do_normalize_ir_structs:
            assert 'ir' in key
            assert row is not None
        
        try:
            k = self.get_asm_key(key, asm_key, compiler, fPIC=fPIC)
            if not row:
                return k
            if "asm" in row and "target" in row["asm"]:
                try:
                    asm_idx = row['asm']['target'].index(k)
                    res = row['asm']['code'][asm_idx]
                except ValueError as e:
                    if not fPIC:
                        raise ValueError(e)
                    else:
                        k = self.get_asm_key(key, asm_key, compiler, fPIC=False)
                        if not row:
                            return k
                        try:
                            asm_idx = row['asm']['target'].index(k)
                            res = row['asm']['code'][asm_idx]
                        except ValueError:
                            return ''
            else:
                if k in row:
                    res = row[k]
                elif key in row:
                    res = row[key]
                else:
                    return ''
        except Exception as e:
            return ''
            
        if 'ir' in key and do_normalize_ir_structs:
            res = normalize_structs(res)
        return res


    def get_par_data(self, row, pair, asm_key='angha', fPIC=False, tokenize_ids=True, do_normalize_ir_structs=False):
        # if row is optional, only return asm full keys
        assert asm_key in ['real', 'angha']
        if asm_key == 'angha':
            io_key = 'pruned_synth_io_pairs'
        else:
            io_key = 'real_io_pairs'

        target_l = pair.split('-')[1]
        source_compiler = 'clang' if ('clang' in pair.split('-')[0].replace(target_l, '') or 'ir' in pair.split('-')[0].replace(target_l, '')) else 'gcc'
        target_compiler = 'clang' if (source_compiler == 'clang' or 'clang' in target_l or 'ir' in target_l) else 'gcc'
        pair = pair.replace('clang_', '')

        if pair == 'c_s-s':
            source = row['func_def']
            target = self.get_asm('x86', row, asm_key, compiler=target_compiler, fPIC=fPIC)
        elif pair == 'c_arm-arm':
            source = row['func_def']
            target = self.get_asm('arm', row, asm_key, compiler=target_compiler, fPIC=fPIC)
        elif pair == 'arm_armopt-armopt':
            source = self.get_asm('arm', row, asm_key)
            target = self.get_asm('armopt', row, asm_key,compiler=target_compiler,fPIC=fPIC)
        elif pair == 's_c-c':
            target = row['func_def']
            source = self.get_asm('x86', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 's_opt-opt':
            source = self.get_asm('x86', row, asm_key, compiler=source_compiler, fPIC=fPIC)
            target = self.get_asm('opt', row, asm_key, compiler=target_compiler, fPIC=fPIC)
        elif pair == 's_arm-arm':
            source = self.get_asm('x86', row, asm_key, compiler=source_compiler, fPIC=fPIC)
            target = self.get_asm('arm', row, asm_key, compiler=target_compiler, fPIC=fPIC)
        elif pair == 'opt3_c-c':
            target = row['func_def']
            source = self.get_asm('opt3', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'arm_c-c':
            target = row['func_def']
            source = self.get_asm('arm', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'arm_opts_c-c':
            target = row['func_def']
            source = self.get_asm('arm_opts', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'arm_opt3_c-c':
            target = row['func_def']
            source = self.get_asm('arm_opt3', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'opt3_s-s':
            target = self.get_asm('x86', row, asm_key, compiler=target_compiler, fPIC=fPIC)
            source = self.get_asm('opt3', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 's_s-s':
            target = self.get_asm('x86', row, asm_key, compiler=target_compiler, fPIC=fPIC)
            source = self.get_asm('x86', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 's_opt3-opt3':
            target = self.get_asm('opt3', row, asm_key, compiler=target_compiler, fPIC=fPIC)
            source = self.get_asm('x86', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'riscv_ir-ir':
            target = self.get_asm('ir', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('riscv', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'riscv_opts_ir_opts-ir_opts':
            target = self.get_asm('ir', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('riscv_opts', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'riscv_opt3_ir_opt3-ir_opt3':
            target = self.get_asm('ir_opt3', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('riscv_opt3', row, asm_key, compiler=source_compiler, fPIC=fPIC)

        elif pair == 's_ir-ir':
            target = self.get_asm('ir', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('x86', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'opts_ir_opts-ir_opts':
            target = self.get_asm('ir_opts', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('opts', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'opt3_ir_opt3-ir_opt3':
            target = self.get_asm('ir_opt3', row, asm_key, compiler=target_compiler,fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('opt3', row, asm_key, compiler=source_compiler, fPIC=fPIC)

        elif pair == 'opt3_ir-ir':
            target = self.get_asm('ir', row, asm_key, compiler=target_compiler,fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('opt3', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'opt3_ir_optz-ir_optz':
            target = self.get_asm('ir_optz', row, asm_key, compiler=target_compiler,fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('opt3', row, asm_key, compiler=source_compiler, fPIC=fPIC)

        elif pair == 'riscv_ir-ir':
            target = self.get_asm('ir', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('riscv', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'riscv_opts_ir_opts-ir_opts':
            target = self.get_asm('ir', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('riscv_opts', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'riscv_opt3_ir_opt3-ir_opt3':
            target = self.get_asm('ir_opt3', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('riscv_opt3', row, asm_key, compiler=source_compiler, fPIC=fPIC)

        elif pair == 'arm_ir-ir':
            target = self.get_asm('ir', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('arm', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'arm_opts_ir_opt3-ir_opts':
            target = self.get_asm('ir', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('arm_opts', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'arm_opt3_ir_opt3-ir_opt3':
            target = self.get_asm('ir_opt3', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('arm_opt3', row, asm_key, compiler=source_compiler,fPIC=fPIC)

        elif pair == 'arm_opt3_ir-ir':
            target = self.get_asm('ir', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('arm_opt3', row, asm_key, compiler=source_compiler, fPIC=fPIC)

        elif pair == 'riscv_opt3_ir-ir':
            target = self.get_asm('ir', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('riscv_opt3', row, asm_key, compiler=source_compiler, fPIC=fPIC)


        elif pair == 'arm32_ir-ir':
            target = self.get_asm('ir', row, asm_key, compiler=target_compiler,  fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('arm32', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'arm32_opts_ir_opts-ir_opts':
            target = self.get_asm('ir', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('arm32_opts', row, asm_key, compiler=source_compiler, fPIC=fPIC)
        elif pair == 'arm32_opt3_ir_opt3-ir_opt3':
            target = self.get_asm('ir_opt3', row, asm_key, compiler=target_compiler, fPIC=fPIC, do_normalize_ir_structs=do_normalize_ir_structs)
            source = self.get_asm('arm32_opt3', row, asm_key, compiler=source_compiler, fPIC=fPIC)

        elif pair == 'opt3_arm_opt3-arm_opt3':
            target = self.get_asm('arm_opt3', row, asm_key, compiler=target_compiler,fPIC=fPIC)
            source = self.get_asm('opt3', row, asm_key, compiler=target_compiler,fPIC=fPIC)

        elif pair == 'arm_opt3_ir_optz-ir_optz':
            target = self.get_asm('ir_optz', row, asm_key, compiler=target_compiler, fPIC=fPIC)
            source = self.get_asm('arm_opt3', row, asm_key, compiler=target_compiler, fPIC=fPIC)
        elif pair == 'riscv_opt3_ir_optz-ir_optz':
            target = self.get_asm('ir_optz', row, asm_key, compiler=target_compiler, fPIC=fPIC)
            source = self.get_asm('riscv_opt3', row, asm_key, compiler=target_compiler, fPIC=fPIC)
        elif 'io' in pair:
            raise ValueError
            # irrelevant for Forklift
        else:
            raise ValueError(pair)
        tokenized_source, tokenized_target = None, None
        if self.tokenizer:
            tokenized_source, tokenized_target = self.tokenize(source, target, pair, ids=tokenize_ids)

        return source, target, tokenized_source, tokenized_target
