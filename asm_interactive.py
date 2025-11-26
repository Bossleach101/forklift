from forklift.evaluator import Evaluator, Config
import os
from forklift.asm import AsmAdder, FuncDataclass
from forklift.utils import normalize_structs, InferenceDataset
import math

# Use the SIMPLE example first
SAMPLE_WITH_ASM_SIMPLE = dict(
    func_def='',
    deps='',
    fname='f',
    asm_code='''
	.globl	f
	.type	f, @function
f:
.LFB6:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$144, %rsp
	movq	%rdi, -136(%rbp)
	movl	%esi, -140(%rbp)
	movl	%edx, -144(%rbp)
	leaq	.L13(%rip), %rax
	movq	%rax, -128(%rbp)
	leaq	.L13(%rip), %rax
	movq	%rax, -120(%rbp)
	leaq	.L14(%rip), %rax
	movq	%rax, -112(%rbp)
	leaq	.L15(%rip), %rax
	movq	%rax, -104(%rbp)
	leaq	.L13(%rip), %rax
	movq	%rax, -96(%rbp)
	leaq	.L16(%rip), %rax
	movq	%rax, -88(%rbp)
	leaq	.L17(%rip), %rax
	movq	%rax, -80(%rbp)
	leaq	.L18(%rip), %rax
	movq	%rax, -72(%rbp)
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-8(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-8(%rbp), %rax
	movq	%rax, _TIG_iO_XeMX_1_f_1_opaque_list_1(%rip)
	call	rand@PLT
	movl	%eax, -12(%rbp)
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	$0, 16(%rax)
	movq	_TIG_iO_XeMX_1_f_1_opaque_list_1(%rip), %rax
	movq	8(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	_TIG_iO_XeMX_1_f_1_opaque_list_1(%rip), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, (%rax)
	movq	_TIG_iO_XeMX_1_f_1_opaque_list_1(%rip), %rax
	movq	8(%rax), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	_TIG_iO_XeMX_1_f_1_opaque_list_1(%rip), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 8(%rax)
	call	rand@PLT
	movl	%eax, -28(%rbp)
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movl	-28(%rbp), %edx
	movl	%edx, 16(%rax)
	movq	_TIG_iO_XeMX_1_f_1_opaque_list_1(%rip), %rax
	movq	8(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	_TIG_iO_XeMX_1_f_1_opaque_list_1(%rip), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, (%rax)
	movq	_TIG_iO_XeMX_1_f_1_opaque_list_1(%rip), %rax
	movq	8(%rax), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	_TIG_iO_XeMX_1_f_1_opaque_list_1(%rip), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	_TIG_iO_XeMX_1_f_1_opaque_list_1(%rip), %rax
	movq	8(%rax), %rax
	movq	%rax, _TIG_iO_XeMX_1_f_1_opaque_ptr_1(%rip)
	movq	_TIG_iO_XeMX_1_f_1_opaque_ptr_1(%rip), %rax
	movq	%rax, _TIG_iO_XeMX_1_f_1_opaque_ptr_2(%rip)
	nop
	nop
	movq	$5, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	-128(%rbp,%rax,8), %rax
	nop
	jmp	*%rax
.L15:
	leaq	.L14(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L20:
	.byte 0x1,0x0,0x0,0x0,0x44,0x89,0xe0
	nop
.L17:
	movl	-52(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-136(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	xorl	-140(%rbp), %eax
	movl	%eax, %edx
	movl	-52(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-136(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	andl	-140(%rbp), %eax
	leal	(%rax,%rax), %ecx
	movl	-52(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rsi
	movq	-136(%rbp), %rax
	addq	%rsi, %rax
	addl	%ecx, %edx
	movl	%edx, (%rax)
	addl	$1, -52(%rbp)
	leaq	.L14(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L21:
	.byte 0x1d
	nop
.L16:
	movl	$0, -52(%rbp)
	leaq	.L14(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L22:
	.byte 0x6,0xd4,0xf6,0x5a,0xd0
	nop
.L13:
	movq	_TIG_iO_XeMX_1_f_1_opaque_ptr_1(%rip), %rdx
	movq	_TIG_iO_XeMX_1_f_1_opaque_ptr_2(%rip), %rax
	cmpq	%rax, %rdx
	jne	.L23
	leaq	.L17(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L24:
	.byte 0x36,0x17,0xda,0xc2
	jmp	.L18
.L23:
	leaq	.L15(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L26:
	.byte 0x20,0x27,0x1b,0x7d,0x5e,0x80
	nop
.L18:
	movl	-144(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-136(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	jmp	.L12
.L14:
	movl	-52(%rbp), %eax
	subl	-144(%rbp), %eax
	movl	%eax, %edx
	movl	-52(%rbp), %eax
	xorl	-144(%rbp), %eax
	movl	%eax, %ecx
	movl	-52(%rbp), %eax
	subl	-144(%rbp), %eax
	xorl	-52(%rbp), %eax
	andl	%ecx, %eax
	xorl	%edx, %eax
	testl	%eax, %eax
	jns	.L28
	leaq	.L13(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L29:
	.byte 0xfd
	jmp	.L32
.L28:
	leaq	.L18(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L31:
	.byte 0x0,0xc7,0x5
	nop
.L32:
.L12:
	leave
	ret
'''
)

SAMPLE_BYTECODE_INTERPRETER = dict(
    func_def='',
    deps='',
    fname='f',
    asm_code='''
    .globl	f
    .type	f, @function
f:
    pushq	%rbp
    movq	%rsp, %rbp
    subq	$472, %rsp
    movq	%rdi, -584(%rbp)
    movl	%esi, -588(%rbp)
    movl	%edx, -592(%rbp)
    leaq	-560(%rbp), %rax
    addq	$24, %rax
    movq	%rax, -576(%rbp)
    leaq	_TIG_VZ_Rnwo_1_f_$array(%rip), %rax
    movq	%rax, -568(%rbp)
    leaq	-560(%rbp), %rax
    addq	$16, %rax
    movq	%rax, -8(%rbp)
    leaq	-592(%rbp), %rdx
    movq	-8(%rbp), %rax
    movq	%rdx, (%rax)
    leaq	-560(%rbp), %rax
    addq	$8, %rax
    movq	%rax, -8(%rbp)
    leaq	-588(%rbp), %rdx
    movq	-8(%rbp), %rax
    movq	%rdx, (%rax)
    leaq	-560(%rbp), %rax
    movq	%rax, -8(%rbp)
    leaq	-584(%rbp), %rdx
    movq	-8(%rbp), %rax
    movq	%rdx, (%rax)
.L29:
    movq	-568(%rbp), %rax
    movzbl	(%rax), %eax
    movzbl	%al, %eax
    cmpl	$252, %eax
    je	.L10
    cmpl	$252, %eax
    jg	.L29
    cmpl	$193, %eax
    je	.L12
    cmpl	$193, %eax
    jg	.L29
    cmpl	$169, %eax
    je	.L13
    cmpl	$169, %eax
    jg	.L29
    cmpl	$153, %eax
    je	.L14
    cmpl	$153, %eax
    jg	.L29
    cmpl	$145, %eax
    je	.L15
    cmpl	$145, %eax
    jg	.L29
    cmpl	$137, %eax
    je	.L16
    cmpl	$137, %eax
    jg	.L29
    cmpl	$114, %eax
    je	.L17
    cmpl	$114, %eax
    jg	.L29
    cmpl	$112, %eax
    je	.L18
    cmpl	$112, %eax
    jg	.L29
    cmpl	$107, %eax
    je	.L19
    cmpl	$107, %eax
    jg	.L29
    cmpl	$71, %eax
    je	.L20
    cmpl	$71, %eax
    jg	.L29
    cmpl	$53, %eax
    je	.L21
    cmpl	$53, %eax
    jg	.L29
    cmpl	$43, %eax
    je	.L22
    cmpl	$43, %eax
    jg	.L29
    cmpl	$38, %eax
    je	.L23
    cmpl	$38, %eax
    jg	.L29
    cmpl	$22, %eax
    je	.L24
    cmpl	$37, %eax
    je	.L25
    jmp	.L29
.L13:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-568(%rbp), %rdx
    movq	-568(%rbp), %rax
    movl	(%rax), %eax
    cltq
    addq	%rdx, %rax
    movq	%rax, -568(%rbp)
    jmp	.L11
.L25:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    jmp	.L31
.L17:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    subq	$16, %rax
    movq	(%rax), %rdx
    movq	-576(%rbp), %rax
    movq	(%rax), %rax
    movq	-576(%rbp), %rcx
    subq	$16, %rcx
    imulq	%rdx, %rax
    movq	%rax, (%rcx)
    movq	-576(%rbp), %rax
    subq	$16, %rax
    movq	%rax, -576(%rbp)
    jmp	.L11
.L21:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    movl	(%rax), %edx
    movq	-576(%rbp), %rax
    movslq	%edx, %rdx
    movq	%rdx, (%rax)
    jmp	.L11
.L19:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    subq	$16, %rax
    movl	(%rax), %edx
    movq	-576(%rbp), %rax
    movl	(%rax), %eax
    cmpl	%eax, %edx
    setl	%cl
    movq	-576(%rbp), %rax
    leaq	-16(%rax), %rdx
    movzbl	%cl, %eax
    movl	%eax, (%rdx)
    movq	-576(%rbp), %rax
    subq	$16, %rax
    movq	%rax, -576(%rbp)
    jmp	.L11
.L14:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    movq	(%rax), %rdx
    movq	-576(%rbp), %rax
    movl	(%rdx), %edx
    movl	%edx, (%rax)
    jmp	.L11
.L18:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-568(%rbp), %rax
    movl	(%rax), %eax
    movslq	%eax, %rdx
    movq	-576(%rbp), %rax
    addq	$16, %rax
    leaq	-560(%rbp), %rcx
    addq	%rcx, %rdx
    movq	%rdx, (%rax)
    movq	-576(%rbp), %rax
    addq	$16, %rax
    movq	%rax, -576(%rbp)
    movq	-568(%rbp), %rax
    addq	$4, %rax
    movq	%rax, -568(%rbp)
    jmp	.L11
.L15:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-568(%rbp), %rax
    movq	-576(%rbp), %rdx
    addq	$16, %rdx
    movl	(%rax), %eax
    movl	%eax, (%rdx)
    movq	-576(%rbp), %rax
    addq	$16, %rax
    movq	%rax, -576(%rbp)
    movq	-568(%rbp), %rax
    addq	$4, %rax
    movq	%rax, -568(%rbp)
    jmp	.L11
.L12:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    leaq	-16(%rax), %rdx
    movq	-576(%rbp), %rax
    movq	(%rax), %rax
    movl	(%rdx), %edx
    movl	%edx, (%rax)
    movq	-576(%rbp), %rax
    subq	$32, %rax
    movq	%rax, -576(%rbp)
    jmp	.L11
.L10:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    movq	(%rax), %rdx
    movq	-576(%rbp), %rax
    movq	(%rdx), %rdx
    movq	%rdx, (%rax)
    jmp	.L11
.L24:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    movq	(%rax), %rcx
    movq	-576(%rbp), %rax
    subq	$16, %rax
    movq	(%rax), %rdx
    movq	-576(%rbp), %rax
    subq	$16, %rax
    addq	%rcx, %rdx
    movq	%rdx, (%rax)
    movq	-576(%rbp), %rax
    subq	$16, %rax
    movq	%rax, -576(%rbp)
    jmp	.L11
.L22:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-568(%rbp), %rax
    movl	(%rax), %eax
    cltq
    leaq	-560(%rbp), %rdx
    addq	%rdx, %rax
    movq	(%rax), %rdx
    movq	-576(%rbp), %rax
    addq	$16, %rax
    movq	%rdx, (%rax)
    movq	-576(%rbp), %rax
    addq	$16, %rax
    movq	%rax, -576(%rbp)
    movq	-568(%rbp), %rax
    addq	$4, %rax
    movq	%rax, -568(%rbp)
    jmp	.L11
.L16:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-568(%rbp), %rax
    movq	-576(%rbp), %rdx
    addq	$16, %rdx
    movq	(%rax), %rax
    movq	%rax, (%rdx)
    movq	-576(%rbp), %rax
    addq	$16, %rax
    movq	%rax, -576(%rbp)
    movq	-568(%rbp), %rax
    addq	$8, %rax
    movq	%rax, -568(%rbp)
    jmp	.L11
.L20:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    movl	(%rax), %eax
    testl	%eax, %eax
    je	.L27
    movq	-568(%rbp), %rdx
    movq	-568(%rbp), %rax
    movl	(%rax), %eax
    cltq
    addq	%rdx, %rax
    movq	%rax, -568(%rbp)
    jmp	.L28
.L27:
    movq	-568(%rbp), %rax
    addq	$4, %rax
    movq	%rax, -568(%rbp)
.L28:
    movq	-576(%rbp), %rax
    subq	$16, %rax
    movq	%rax, -576(%rbp)
    jmp	.L11
.L23:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    subq	$16, %rax
    movl	(%rax), %ecx
    movq	-576(%rbp), %rax
    movl	(%rax), %edx
    movq	-576(%rbp), %rax
    subq	$16, %rax
    addl	%ecx, %edx
    movl	%edx, (%rax)
    movq	-576(%rbp), %rax
    subq	$16, %rax
    movq	%rax, -576(%rbp)
    nop
.L11:
    jmp	.L29
.L31:
    leave
    ret
'''
)



DIRECTION = 'clang_opt3_ir_optz-ir_optz'

MODELS = {'clang_opt3_ir_optz-ir_optz': 'jordiae/clang_opt3_ir_optz-ir_optz-2024-01-15-0959-e1bf-bc2b'}


class InferenceDatasetFromAsm:
    """Modified version that accepts assembly directly"""
    def __init__(self, data, direction):
        self.data = data
        self.direction = direction
        
    def __iter__(self):
        for instance in self.data:
            source_key = 'real_clang_x86_O3'
            target_key = 'real_clang_ir_Oz'
            
            asm_code = instance.get('asm_code', '')
            
            print(f"\n=== DEBUG InferenceDatasetFromAsm ===")
            print(f"Assembly code length: {len(asm_code)} chars")
            print(f"First 300 chars:\n{asm_code[:300]}")
            print(f"Source key: {source_key}")
            print(f"Target key: {target_key}")
            
            row = {
                'func_def': instance['func_def'],
                'fname': instance['fname'],
                'deps': instance.get('deps', ''),
                'real_deps': instance.get('deps', ''),
                'func_head_types': '',
                'asm': {
                    'target': [source_key, target_key],
                    'code': [asm_code, None]
                }
            }
            
            print(f"Row structure: {list(row.keys())}")
            print(f"ASM targets: {row['asm']['target']}")
            print(f"ASM code[0] length: {len(row['asm']['code'][0])}")
            print("=== END DEBUG ===\n")
            
            yield row


def run_with_c_code(sample, batch_size=1):
    """Original function that compiles C code"""
    config = Config(hf_model_path=MODELS[DIRECTION],
                    pairs=[DIRECTION])

    evaluator = Evaluator(config)
    predictions = []
    batch = []
    pair = DIRECTION
    samples = [sample]
    
    for row in InferenceDataset(samples, compilers_keys=['clang_ir_Oz', 'clang_x86_O3']):
        # Debug: check what C code produces
        print(f"DEBUG C: Assembly has {len(row['asm']['code'][0])} chars")
        print(row['asm']['code'][0][:200])
        batch.append((row, pair))
        if len(batch) == batch_size:
            predictions.extend(evaluator.predict_batch(batch))
            batch = []
    if len(batch) > 0:
        predictions.extend(evaluator.predict_batch(batch))

    return predictions[0]


def run_with_assembly(sample, batch_size=1):
    """New function that accepts assembly directly"""
    config = Config(hf_model_path=MODELS[DIRECTION],
                    pairs=[DIRECTION])

    evaluator = Evaluator(config)
    
    print(f"\n=== MODEL LIMITS ===")
    print(f"Max input position embeddings: {evaluator.model.config.max_position_embeddings}")
    print(f"Max output tokens: {config.max_new_tokens}")
    print(f"=== END LIMITS ===\n")
    
    predictions = []
    batch = []
    pair = DIRECTION
    samples = [sample]
    
    for row in InferenceDatasetFromAsm(samples, direction=DIRECTION):
        batch.append((row, pair))
        
        if len(batch) == batch_size:
            # Check token length before processing
            print("\n=== CHECKING SEQUENCE LENGTH ===")
            for r, p in batch:
                tok, len_t = evaluator.data_processor.prepare(r, p, asm_key=evaluator.asm_key, return_target_length=True)
                print(f"Source (assembly) tokens: {len(tok)}")
                print(f"Target estimate: {len_t}")
                print(f"Max allowed input: {evaluator.model.config.max_position_embeddings}")
                
                if len(tok) > evaluator.model.config.max_position_embeddings:
                    print(f"\n❌ ERROR: Input assembly too long!")
                    print(f"   Your assembly: {len(tok)} tokens")
                    print(f"   Model limit: {evaluator.model.config.max_position_embeddings} tokens")
                    print(f"   Need to reduce by: {len(tok) - evaluator.model.config.max_position_embeddings} tokens (~{((len(tok) - evaluator.model.config.max_position_embeddings) / len(tok) * 100):.1f}% reduction)")
                    
                    # Estimate assembly lines to remove
                    asm_lines = r['asm']['code'][0].count('\n')
                    lines_to_remove = math.ceil((len(tok) - evaluator.model.config.max_position_embeddings) / len(tok) * asm_lines)
                    print(f"   Approximate assembly lines to remove: {lines_to_remove} out of {asm_lines}")
                    return None
                else:
                    print(f"✓ Input size OK ({len(tok)} / {evaluator.model.config.max_position_embeddings} tokens used)")
            print("=== END CHECK ===\n")
            
            try:
                predictions.extend(evaluator.predict_batch(batch))
            except Exception as e:
                print(f"ERROR in predict_batch: {e}")
                import traceback
                traceback.print_exc()
                raise
            batch = []
            
    if len(batch) > 0:
        # Same check for remaining batch
        for r, p in batch:
            tok, len_t = evaluator.data_processor.prepare(r, p, asm_key=evaluator.asm_key, return_target_length=True)
            if len(tok) > evaluator.model.config.max_position_embeddings:
                print(f"❌ Input too long: {len(tok)} > {evaluator.model.config.max_position_embeddings}")
                return None
        
        try:
            predictions.extend(evaluator.predict_batch(batch))
        except Exception as e:
            print(f"ERROR in predict_batch: {e}")
            import traceback
            traceback.print_exc()
            raise

    return predictions[0] if predictions else None


def compare_assembly_formats(c_sample, asm_sample):
    """Compare what C compilation produces vs what we're providing"""
    print("\n=== COMPARING ASSEMBLY FORMATS ===")
    
    # Get C-generated assembly
    for row in InferenceDataset([c_sample], compilers_keys=['clang_ir_Oz', 'clang_x86_O3']):
        c_asm = row['asm']['code'][0]
        print(f"C-generated assembly ({len(c_asm)} chars):")
        print(c_asm[:500])
        print("\n")
        
        # Analyze format
        c_lines = c_asm.split('\n')
        print(f"Total lines: {len(c_lines)}")
        print(f"Lines starting with tab: {sum(1 for l in c_lines if l.startswith('\t'))}")
        print(f"Lines starting with dot: {sum(1 for l in c_lines if l.strip().startswith('.'))}")
        print(f"Empty lines: {sum(1 for l in c_lines if not l.strip())}")
        break
    
    # Get our provided assembly
    for row in InferenceDatasetFromAsm([asm_sample], direction=DIRECTION):
        our_asm = row['asm']['code'][0]
        print(f"\nOur assembly ({len(our_asm)} chars):")
        print(our_asm[:500])
        print("\n")
        
        # Analyze format
        our_lines = our_asm.split('\n')
        print(f"Total lines: {len(our_lines)}")
        print(f"Lines starting with tab: {sum(1 for l in our_lines if l.startswith('\t'))}")
        print(f"Lines starting with dot: {sum(1 for l in our_lines if l.strip().startswith('.'))}")
        print(f"Empty lines: {sum(1 for l in our_lines if not l.strip())}")
        break
    
    print("=== END COMPARISON ===\n")


# ==============================================================================
# PRE-CONFIGURED TEST CASES FOR TIGRESS BYTECODE OPCODES
# ==============================================================================

# Test 1: Memory Operations Only (2 opcodes)
# Opcodes: 252 (dereference), 153 (load from memory)
BYTECODE_MEMORY_OPS = dict(
    func_def='',
    deps='',
    fname='f',
    asm_code='''
    .globl	f
    .type	f, @function
f:
    pushq	%rbp
    movq	%rsp, %rbp
    subq	$472, %rsp
    movq	%rdi, -584(%rbp)
    movl	%esi, -588(%rbp)
    movl	%edx, -592(%rbp)
    leaq	-560(%rbp), %rax
    addq	$24, %rax
    movq	%rax, -576(%rbp)
    leaq	_TIG_VZ_Rnwo_1_f_$array(%rip), %rax
    movq	%rax, -568(%rbp)
.L29:
    movq	-568(%rbp), %rax
    movzbl	(%rax), %eax
    movzbl	%al, %eax
    cmpl	$252, %eax
    je	.L10
    cmpl	$153, %eax
    je	.L14
    jmp	.L29
.L14:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    movq	(%rax), %rdx
    movq	-576(%rbp), %rax
    movl	(%rdx), %edx
    movl	%edx, (%rax)
    jmp	.L11
.L10:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    movq	(%rax), %rdx
    movq	-576(%rbp), %rax
    movq	(%rdx), %rdx
    movq	%rdx, (%rax)
    jmp	.L11
.L11:
    jmp	.L29
    leave
    ret
'''
)

# Test 2: Memory + Control Flow (3 opcodes)
# Opcodes: 252 (dereference), 193 (store), 169 (relative jump)
BYTECODE_MEM_CONTROL = dict(
    func_def='',
    deps='',
    fname='f',
    asm_code='''
    .globl	f
    .type	f, @function
f:
    pushq	%rbp
    movq	%rsp, %rbp
    subq	$472, %rsp
    movq	%rdi, -584(%rbp)
    movl	%esi, -588(%rbp)
    movl	%edx, -592(%rbp)
    leaq	-560(%rbp), %rax
    addq	$24, %rax
    movq	%rax, -576(%rbp)
    leaq	_TIG_VZ_Rnwo_1_f_$array(%rip), %rax
    movq	%rax, -568(%rbp)
.L29:
    movq	-568(%rbp), %rax
    movzbl	(%rax), %eax
    movzbl	%al, %eax
    cmpl	$252, %eax
    je	.L10
    cmpl	$193, %eax
    je	.L12
    cmpl	$169, %eax
    je	.L13
    jmp	.L29
.L13:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-568(%rbp), %rdx
    movq	-568(%rbp), %rax
    movl	(%rax), %eax
    cltq
    addq	%rdx, %rax
    movq	%rax, -568(%rbp)
    jmp	.L11
.L12:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    leaq	-16(%rax), %rdx
    movq	-576(%rbp), %rax
    movq	(%rax), %rax
    movl	(%rdx), %edx
    movl	%edx, (%rax)
    movq	-576(%rbp), %rax
    subq	$32, %rax
    movq	%rax, -576(%rbp)
    jmp	.L11
.L10:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    movq	(%rax), %rdx
    movq	-576(%rbp), %rax
    movq	(%rdx), %rdx
    movq	%rdx, (%rax)
    jmp	.L11
.L11:
    jmp	.L29
    leave
    ret
'''
)

# Test 3: Stack Operations (4 opcodes)
# Opcodes: 38 (push), 22 (pop), 252 (dereference), 193 (store)
BYTECODE_STACK_OPS = dict(
    func_def='',
    deps='',
    fname='f',
    asm_code='''
    .globl	f
    .type	f, @function
f:
    pushq	%rbp
    movq	%rsp, %rbp
    subq	$472, %rsp
    movq	%rdi, -584(%rbp)
    movl	%esi, -588(%rbp)
    movl	%edx, -592(%rbp)
    leaq	-560(%rbp), %rax
    addq	$24, %rax
    movq	%rax, -576(%rbp)
    leaq	_TIG_VZ_Rnwo_1_f_$array(%rip), %rax
    movq	%rax, -568(%rbp)
    movl	$16843009, %edx
    movl	%edx, -44(%rbp)
    movl	$33686018, %edx
    movl	%edx, -40(%rbp)
    movl	$50529027, %edx
    movl	%edx, -36(%rbp)
    movl	$67372036, %edx
    movl	%edx, -32(%rbp)
.L29:
    movq	-568(%rbp), %rax
    movzbl	(%rax), %eax
    movzbl	%al, %eax
    cmpl	$252, %eax
    je	.L10
    cmpl	$193, %eax
    je	.L12
    cmpl	$38, %eax
    je	.L15
    cmpl	$22, %eax
    je	.L16
    jmp	.L29
.L15:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-568(%rbp), %rax
    movl	(%rax), %edx
    movq	-576(%rbp), %rax
    addq	$16, %rax
    movl	%edx, (%rax)
    movq	-576(%rbp), %rax
    addq	$16, %rax
    movq	%rax, -576(%rbp)
    movq	-568(%rbp), %rax
    addq	$4, %rax
    movq	%rax, -568(%rbp)
    jmp	.L11
.L16:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    movl	(%rax), %edx
    leaq	-592(%rbp), %rcx
    movq	-568(%rbp), %rax
    movl	%edx, (%rcx,%rax)
    movq	-576(%rbp), %rax
    subq	$16, %rax
    movq	%rax, -576(%rbp)
    movq	-568(%rbp), %rax
    addq	$4, %rax
    movq	%rax, -568(%rbp)
    jmp	.L11
.L12:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    leaq	-16(%rax), %rdx
    movq	-576(%rbp), %rax
    movq	(%rax), %rax
    movl	(%rdx), %edx
    movl	%edx, (%rax)
    movq	-576(%rbp), %rax
    subq	$32, %rax
    movq	%rax, -576(%rbp)
    jmp	.L11
.L10:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    movq	(%rax), %rdx
    movq	-576(%rbp), %rax
    movq	(%rdx), %rdx
    movq	%rdx, (%rax)
    jmp	.L11
.L11:
    jmp	.L29
    leave
    ret
'''
)

# Test 4: Minimal (Single Opcode) - For baseline testing
# Opcode: 252 (dereference only)
BYTECODE_SINGLE_OPCODE = dict(
    func_def='',
    deps='',
    fname='f',
    asm_code='''
    .globl	f
    .type	f, @function
f:
    pushq	%rbp
    movq	%rsp, %rbp
    subq	$472, %rsp
    movq	%rdi, -584(%rbp)
    movl	%esi, -588(%rbp)
    movl	%edx, -592(%rbp)
    leaq	-560(%rbp), %rax
    addq	$24, %rax
    movq	%rax, -576(%rbp)
    leaq	_TIG_VZ_Rnwo_1_f_$array(%rip), %rax
    movq	%rax, -568(%rbp)
.L29:
    movq	-568(%rbp), %rax
    movzbl	(%rax), %eax
    movzbl	%al, %eax
    cmpl	$252, %eax
    je	.L10
    jmp	.L29
.L10:
    movq	-568(%rbp), %rax
    addq	$1, %rax
    movq	%rax, -568(%rbp)
    movq	-576(%rbp), %rax
    movq	(%rax), %rdx
    movq	-576(%rbp), %rax
    movq	(%rdx), %rdx
    movq	%rdx, (%rax)
    jmp	.L11
.L11:
    jmp	.L29
    leave
    ret
'''
)

# ==============================================================================
# TEST RUNNER WITH ALL CONFIGURATIONS
# ==============================================================================

def run_test_suite():
    """Run all bytecode test configurations"""
    
    test_cases = [
        ("Single Opcode (Baseline)", BYTECODE_SINGLE_OPCODE, 
         "Tests: Opcode 252 (dereference pointer)"),
        
        ("Memory Operations", BYTECODE_MEMORY_OPS,
         "Tests: Opcodes 252 (dereference), 153 (load from memory)"),
        
        ("Memory + Control Flow", BYTECODE_MEM_CONTROL,
         "Tests: Opcodes 252 (dereference), 193 (store), 169 (relative jump)"),
        
        ("Stack Operations", BYTECODE_STACK_OPS,
         "Tests: Opcodes 38 (push), 22 (pop), 252 (dereference), 193 (store)"),
    ]
    
    results = {}
    
    for name, test_case, description in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST: {name}")
        print(f"{'='*70}")
        print(f"Description: {description}\n")
        
        predicted = run_with_assembly(test_case)
        
        if predicted:
            print(f"\n✅ SUCCESS - Model generated output:")
            print("-" * 70)
            for idx, lifted in enumerate(predicted):
                print(lifted)
                if idx < len(predicted) - 1:
                    print('_____')
            results[name] = "SUCCESS"
        else:
            print(f"\n❌ FAILED - Assembly too large or other error")
            results[name] = "FAILED"
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    for name, status in results.items():
        symbol = "✅" if status == "SUCCESS" else "❌"
        print(f"{symbol} {name}: {status}")
    
    return results

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # Compare formats first
    # compare_assembly_formats(SAMPLE_C, BYTECODE_MEMORY_OPS)
    
    # Run the full test suite
    # print("\n\n" + "="*70)
    # print("STARTING TIGRESS BYTECODE TEST SUITE")
    # print("="*70)
    
    # run_test_suite()

    predicted = run_with_assembly(SAMPLE_WITH_ASM_SIMPLE)
    for idx, lifted in enumerate(predicted):
                print(lifted)
                if idx < len(predicted) - 1:
                    print('_____')