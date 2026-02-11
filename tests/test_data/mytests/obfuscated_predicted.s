	.text
	.file	"obfuscated_predicted.ll"
	.globl	i64                             # -- Begin function i64
	.p2align	4, 0x90
	.type	i64,@function
i64:                                    # @i64
	.cfi_startproc
# %bb.0:
	movq	$-64, %rax
	testq	%rax, %rax
	je	.LBB0_3
	.p2align	4, 0x90
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	movl	$0, .str+64(%rax)
	addq	$4, %rax
	testq	%rax, %rax
	jne	.LBB0_2
.LBB0_3:
	retq
.Lfunc_end0:
	.size	i64, .Lfunc_end0-i64
	.cfi_endproc
                                        # -- End function
	.hidden	.str
	.section	".note.GNU-stack","",@progbits
