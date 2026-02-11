	.text
	.file	"bubble_sort.c"
	.globl	f                               // -- Begin function f
	.p2align	2
	.type	f,@function
f:                                      // @f
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	str	x0, [sp, #24]
	str	w1, [sp, #20]
	str	wzr, [sp, #16]
	b	.LBB0_1
.LBB0_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_3 Depth 2
	ldr	w8, [sp, #16]
	ldr	w9, [sp, #20]
	subs	w9, w9, #1
	subs	w8, w8, w9
	b.ge	.LBB0_10
	b	.LBB0_2
.LBB0_2:                                //   in Loop: Header=BB0_1 Depth=1
	str	wzr, [sp, #12]
	b	.LBB0_3
.LBB0_3:                                //   Parent Loop BB0_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	w8, [sp, #12]
	ldr	w9, [sp, #20]
	ldr	w10, [sp, #16]
	subs	w9, w9, w10
	subs	w9, w9, #1
	subs	w8, w8, w9
	b.ge	.LBB0_8
	b	.LBB0_4
.LBB0_4:                                //   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #24]
	ldrsw	x9, [sp, #12]
	ldr	w8, [x8, x9, lsl #2]
	ldr	x9, [sp, #24]
	ldr	w10, [sp, #12]
	add	w10, w10, #1
	ldr	w9, [x9, w10, sxtw #2]
	subs	w8, w8, w9
	b.le	.LBB0_6
	b	.LBB0_5
.LBB0_5:                                //   in Loop: Header=BB0_3 Depth=2
	ldr	x8, [sp, #24]
	ldrsw	x9, [sp, #12]
	ldr	w8, [x8, x9, lsl #2]
	str	w8, [sp, #8]
	ldr	x8, [sp, #24]
	ldr	w9, [sp, #12]
	add	w9, w9, #1
	ldr	w8, [x8, w9, sxtw #2]
	ldr	x9, [sp, #24]
	ldrsw	x10, [sp, #12]
	str	w8, [x9, x10, lsl #2]
	ldr	w8, [sp, #8]
	ldr	x9, [sp, #24]
	ldr	w10, [sp, #12]
	add	w10, w10, #1
	str	w8, [x9, w10, sxtw #2]
	b	.LBB0_6
.LBB0_6:                                //   in Loop: Header=BB0_3 Depth=2
	b	.LBB0_7
.LBB0_7:                                //   in Loop: Header=BB0_3 Depth=2
	ldr	w8, [sp, #12]
	add	w8, w8, #1
	str	w8, [sp, #12]
	b	.LBB0_3
.LBB0_8:                                //   in Loop: Header=BB0_1 Depth=1
	b	.LBB0_9
.LBB0_9:                                //   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [sp, #16]
	add	w8, w8, #1
	str	w8, [sp, #16]
	b	.LBB0_1
.LBB0_10:
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
