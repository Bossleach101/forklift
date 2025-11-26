	.file	"output1.c"
	.text
	.globl	_TIG_iO_XeMX_1_f_1_opaque_list_1
	.bss
	.align 8
	.type	_TIG_iO_XeMX_1_f_1_opaque_list_1, @object
	.size	_TIG_iO_XeMX_1_f_1_opaque_list_1, 8
_TIG_iO_XeMX_1_f_1_opaque_list_1:
	.zero	8
	.globl	_TIG_iO_XeMX_1_f_1_opaque_ptr_2
	.align 8
	.type	_TIG_iO_XeMX_1_f_1_opaque_ptr_2, @object
	.size	_TIG_iO_XeMX_1_f_1_opaque_ptr_2, 8
_TIG_iO_XeMX_1_f_1_opaque_ptr_2:
	.zero	8
	.globl	_TIG_IZ_XeMX_envp
	.align 8
	.type	_TIG_IZ_XeMX_envp, @object
	.size	_TIG_IZ_XeMX_envp, 8
_TIG_IZ_XeMX_envp:
	.zero	8
	.globl	_TIG_iO_XeMX_1_f_1_opaque_ptr_1
	.align 8
	.type	_TIG_iO_XeMX_1_f_1_opaque_ptr_1, @object
	.size	_TIG_iO_XeMX_1_f_1_opaque_ptr_1, 8
_TIG_iO_XeMX_1_f_1_opaque_ptr_1:
	.zero	8
	.globl	_TIG_IZ_XeMX_argv
	.align 8
	.type	_TIG_IZ_XeMX_argv, @object
	.size	_TIG_IZ_XeMX_argv, 8
_TIG_IZ_XeMX_argv:
	.zero	8
	.globl	_TIG_IZ_XeMX_argc
	.align 4
	.type	_TIG_IZ_XeMX_argc, @object
	.size	_TIG_IZ_XeMX_argc, 4
_TIG_IZ_XeMX_argc:
	.zero	4
	.globl	_TIG_iO_XeMX_1_f_1_opaque_list_2
	.align 8
	.type	_TIG_iO_XeMX_1_f_1_opaque_list_2, @object
	.size	_TIG_iO_XeMX_1_f_1_opaque_list_2, 8
_TIG_iO_XeMX_1_f_1_opaque_list_2:
	.zero	8
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movl	$0, _TIG_IZ_XeMX_argc(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_XeMX_argv(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_XeMX_envp(%rip)
	nop
.L4:
	movq	$0, _TIG_iO_XeMX_1_f_1_opaque_list_1(%rip)
	nop
.L5:
	movq	$0, _TIG_iO_XeMX_1_f_1_opaque_list_2(%rip)
	nop
.L6:
	movq	$0, _TIG_iO_XeMX_1_f_1_opaque_ptr_1(%rip)
	nop
.L7:
	movq	$0, _TIG_iO_XeMX_1_f_1_opaque_ptr_2(%rip)
	nop
	nop
.L8:
.L9:
#APP
# 141 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-XeMX--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_XeMX_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_XeMX_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_XeMX_envp(%rip)
	nop
	movl	$1, -32(%rbp)
	movl	$2, -28(%rbp)
	movl	$3, -24(%rbp)
	movl	$4, -20(%rbp)
	movl	$5, -16(%rbp)
	leaq	-32(%rbp), %rax
	movl	$5, %edx
	movl	$10, %esi
	movq	%rax, %rdi
	call	f
	movl	%eax, -4(%rbp)
	movl	-4(%rbp), %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.globl	_TIG_iB_XeMX_2_bf_1
	.type	_TIG_iB_XeMX_2_bf_1, @function
_TIG_iB_XeMX_2_bf_1:
.LFB3:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
#APP
# 26 "f_ground_truth.c" 1
	movq  %rax, 8(%rbp)
# 0 "" 2
#NO_APP
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	_TIG_iB_XeMX_2_bf_1, .-_TIG_iB_XeMX_2_bf_1
	.globl	f
	.type	f, @function
f:
.LFB6:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
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
#APP
# 57 "/usr/include/x86_64-linux-gnu/bits/uintn-identity.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-iO-XeMX-1-f--2
# 0 "" 2
# 61 "/usr/include/x86_64-linux-gnu/bits/uintn-identity.h" 1
	##_ANNOTATION_ATOMICREGION_-TIG-iO-XeMX-1-f--1
# 0 "" 2
#NO_APP
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
#APP
# 92 "/usr/include/x86_64-linux-gnu/bits/uintn-identity.h" 1
	##_ANNOTATION_ATOMICREGION_-TIG-EB-XeMX-4-f--3
# 0 "" 2
#NO_APP
	leaq	.L14(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L20:
#APP
# 95 "/usr/include/x86_64-linux-gnu/bits/uintn-identity.h" 1
	.byte 0x1,0x0,0x0,0x0,0x44,0x89,0xe0
# 0 "" 2
#NO_APP
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
#APP
# 8 "f_ground_truth.c" 1
	##_ANNOTATION_ATOMICREGION_-TIG-EB-XeMX-4-f--4
# 0 "" 2
#NO_APP
	leaq	.L14(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L21:
#APP
# 11 "f_ground_truth.c" 1
	.byte 0x1d
# 0 "" 2
#NO_APP
	nop
.L16:
	movl	$0, -52(%rbp)
#APP
# 7 "f_ground_truth.c" 1
	##_ANNOTATION_ATOMICREGION_-TIG-EB-XeMX-4-f--5
# 0 "" 2
#NO_APP
	leaq	.L14(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L22:
#APP
# 10 "f_ground_truth.c" 1
	.byte 0x6,0xd4,0xf6,0x5a,0xd0
# 0 "" 2
#NO_APP
	nop
.L13:
	movq	_TIG_iO_XeMX_1_f_1_opaque_ptr_1(%rip), %rdx
	movq	_TIG_iO_XeMX_1_f_1_opaque_ptr_2(%rip), %rax
	cmpq	%rax, %rdx
	jne	.L23
#APP
# 16 "f_ground_truth.c" 1
	##_ANNOTATION_ATOMICREGION_-TIG-EB-XeMX-4-f--6
# 0 "" 2
#NO_APP
	leaq	.L17(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L24:
#APP
# 19 "f_ground_truth.c" 1
	.byte 0x36,0x17,0xda,0xc2
# 0 "" 2
#NO_APP
	jmp	.L18
.L23:
#APP
# 24 "f_ground_truth.c" 1
	##_ANNOTATION_ATOMICREGION_-TIG-EB-XeMX-4-f--7
# 0 "" 2
#NO_APP
	leaq	.L15(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L26:
#APP
# 27 "f_ground_truth.c" 1
	.byte 0x20,0x27,0x1b,0x7d,0x5e,0x80
# 0 "" 2
#NO_APP
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
#APP
# 36 "f_ground_truth.c" 1
	##_ANNOTATION_ATOMICREGION_-TIG-EB-XeMX-4-f--8
# 0 "" 2
#NO_APP
	leaq	.L13(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L29:
#APP
# 39 "f_ground_truth.c" 1
	.byte 0xfd
# 0 "" 2
#NO_APP
	jmp	.L32
.L28:
#APP
# 44 "f_ground_truth.c" 1
	##_ANNOTATION_ATOMICREGION_-TIG-EB-XeMX-4-f--9
# 0 "" 2
#NO_APP
	leaq	.L18(%rip), %rax
	movq	%rax, %rdi
	call	_TIG_iB_XeMX_2_bf_1
.L31:
#APP
# 47 "f_ground_truth.c" 1
	.byte 0x0,0xc7,0x5
# 0 "" 2
#NO_APP
	nop
.L32:
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	f, .-f
	.ident	"GCC: (Debian 15.2.0-4) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
