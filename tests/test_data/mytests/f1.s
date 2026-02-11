	.text
	.file	"test1_predicted.ll"
	.globl	i64                             // -- Begin function i64
	.p2align	2
	.type	i64,@function
i64:                                    // @i64
.Li64$local:
	.type	.Li64$local,@function
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	mov	x8, xzr
	str	x8, [sp, #8]                    // 8-byte Folded Spill
	b	.LBB0_1
.LBB0_1:                                // =>This Inner Loop Header: Depth=1
	ldr	x8, [sp, #8]                    // 8-byte Folded Reload
	str	x8, [sp]                        // 8-byte Folded Spill
	subs	x8, x8, #16
	b.eq	.LBB0_3
	b	.LBB0_2
.LBB0_2:                                //   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [sp]                        // 8-byte Folded Reload
	adrp	x10, .str
	add	x10, x10, :lo12:.str
	mov	w9, wzr
	str	w9, [x10, x8, lsl #2]
	add	x8, x8, #1
	str	x8, [sp, #8]                    // 8-byte Folded Spill
	b	.LBB0_1
.LBB0_3:
	add	sp, sp, #16
	ret
.Lfunc_end0:
	.size	i64, .Lfunc_end0-i64
	.size	.Li64$local, .Lfunc_end0-i64
	.cfi_endproc
                                        // -- End function
	.hidden	.str
	.section	".note.GNU-stack","",@progbits
	.addrsig
