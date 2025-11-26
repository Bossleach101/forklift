	.text
	.file	"f_predicted.ll"
	.globl	f                               // -- Begin function f
	.p2align	2
	.type	f,@function
f:                                      // @f
	.cfi_startproc
// %bb.0:
                                        // kill: def $w2 killed $w2 def $x2
	sxtw	x8, w2
	mov	x9, xzr
	cmp	x9, x8
	b.ge	.LBB0_2
.LBB0_1:                                // =>This Inner Loop Header: Depth=1
	ldr	w10, [x0, x9, lsl #2]
	add	w10, w10, w1
	str	w10, [x0, x9, lsl #2]
	add	x9, x9, #1
	cmp	x9, x8
	b.lt	.LBB0_1
.LBB0_2:
	ret
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
