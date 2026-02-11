	.text
	.file	"abs_val.c"
	.globl	f                               // -- Begin function f
	.p2align	2
	.type	f,@function
f:                                      // @f
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	str	w0, [sp, #8]
	ldr	w8, [sp, #8]
	tbz	w8, #31, .LBB0_2
	b	.LBB0_1
.LBB0_1:
	ldr	w9, [sp, #8]
	mov	w8, wzr
	subs	w8, w8, w9
	str	w8, [sp, #12]
	b	.LBB0_3
.LBB0_2:
	ldr	w8, [sp, #8]
	str	w8, [sp, #12]
	b	.LBB0_3
.LBB0_3:
	ldr	w0, [sp, #12]
	add	sp, sp, #16
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
	.cfi_endproc
                                        // -- End function
	.ident	"Debian clang version 19.1.7 (7)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
