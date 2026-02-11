	.text
	.file	"fibonacci.c"
	.globl	f                               // -- Begin function f
	.p2align	2
	.type	f,@function
f:                                      // @f
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	str	w0, [sp, #24]
	ldr	w8, [sp, #24]
	subs	w8, w8, #1
	b.gt	.LBB0_2
	b	.LBB0_1
.LBB0_1:
	ldr	w8, [sp, #24]
	str	w8, [sp, #28]
	b	.LBB0_7
.LBB0_2:
	str	wzr, [sp, #20]
	mov	w8, #1                          // =0x1
	str	w8, [sp, #16]
	mov	w8, #2                          // =0x2
	str	w8, [sp, #12]
	b	.LBB0_3
.LBB0_3:                                // =>This Inner Loop Header: Depth=1
	ldr	w8, [sp, #12]
	ldr	w9, [sp, #24]
	subs	w8, w8, w9
	b.gt	.LBB0_6
	b	.LBB0_4
.LBB0_4:                                //   in Loop: Header=BB0_3 Depth=1
	ldr	w8, [sp, #20]
	ldr	w9, [sp, #16]
	add	w8, w8, w9
	str	w8, [sp, #8]
	ldr	w8, [sp, #16]
	str	w8, [sp, #20]
	ldr	w8, [sp, #8]
	str	w8, [sp, #16]
	b	.LBB0_5
.LBB0_5:                                //   in Loop: Header=BB0_3 Depth=1
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	.LBB0_3
.LBB0_6:
	ldr	w8, [sp, #16]
	str	w8, [sp, #28]
	b	.LBB0_7
.LBB0_7:
	ldr	w0, [sp, #28]
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
	.cfi_endproc
                                        // -- End function
	.ident	"Debian clang version 19.1.7 (7)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
