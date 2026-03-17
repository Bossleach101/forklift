from neurel_deob.training.data import strip_ir_noise

sample_ir = """
; ModuleID = 'f_ground_truth.c'
source_filename = "f_ground_truth.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@.str = external hidden unnamed_addr constant [34 x i8], align 1

; Function Attrs: minsize nofree norecurse nosync nounwind
define dso_local void @f(ptr nocapture noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = add nsw i32 %1, 1
  store i32 %3, ptr %0, align 4
  ret void
}

declare i32 @llvm.smax.i32(i32, i32) #1
"""

result = strip_ir_noise(sample_ir)
print("--- STRIPPED RESULT ---")
print(result)
