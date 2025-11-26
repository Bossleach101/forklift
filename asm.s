f:
        test    edx, edx
        jle     .LBB0_7
        mov     eax, edx
        lea     rcx, [rax - 1]
        movabs  rdx, 4611686018427387903
        and     rdx, rcx
        mov     rcx, rdi
        cmp     rdx, 7
        jb      .LBB0_5
        inc     rdx
        mov     r8, rdx
        and     r8, -8
        lea     rcx, [rdi + 4*r8]
        movd    xmm0, esi
        pshufd  xmm0, xmm0, 0
        xor     r9d, r9d
.LBB0_3:
        movdqu  xmm1, xmmword ptr [rdi + 4*r9]
        movdqu  xmm2, xmmword ptr [rdi + 4*r9 + 16]
        paddd   xmm1, xmm0
        paddd   xmm2, xmm0
        movdqu  xmmword ptr [rdi + 4*r9], xmm1
        movdqu  xmmword ptr [rdi + 4*r9 + 16], xmm2
        add     r9, 8
        cmp     r8, r9
        jne     .LBB0_3
        cmp     rdx, r8
        je      .LBB0_7
.LBB0_5:
        lea     rax, [rdi + 4*rax]
.LBB0_6:
        add     dword ptr [rcx], esi
        add     rcx, 4
        cmp     rcx, rax
        jne     .LBB0_6
.LBB0_7:
        ret