; ModuleID = '/home/leachl/Project/Forklift/forklift/tests/test_data/aarch64_pairs/fibonacci.c'
source_filename = "/home/leachl/Project/Forklift/forklift/tests/test_data/aarch64_pairs/fibonacci.c"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  %8 = load i32, ptr %3, align 4
  %9 = icmp sle i32 %8, 1
  br i1 %9, label %10, label %12

10:                                               ; preds = %1
  %11 = load i32, ptr %3, align 4
  store i32 %11, ptr %2, align 4
  br label %28

12:                                               ; preds = %1
  store i32 0, ptr %4, align 4
  store i32 1, ptr %5, align 4
  store i32 2, ptr %6, align 4
  br label %13

13:                                               ; preds = %23, %12
  %14 = load i32, ptr %6, align 4
  %15 = load i32, ptr %3, align 4
  %16 = icmp sle i32 %14, %15
  br i1 %16, label %17, label %26

17:                                               ; preds = %13
  %18 = load i32, ptr %4, align 4
  %19 = load i32, ptr %5, align 4
  %20 = add nsw i32 %18, %19
  store i32 %20, ptr %7, align 4
  %21 = load i32, ptr %5, align 4
  store i32 %21, ptr %4, align 4
  %22 = load i32, ptr %7, align 4
  store i32 %22, ptr %5, align 4
  br label %23

23:                                               ; preds = %17
  %24 = load i32, ptr %6, align 4
  %25 = add nsw i32 %24, 1
  store i32 %25, ptr %6, align 4
  br label %13, !llvm.loop !6

26:                                               ; preds = %13
  %27 = load i32, ptr %5, align 4
  store i32 %27, ptr %2, align 4
  br label %28

28:                                               ; preds = %26, %10
  %29 = load i32, ptr %2, align 4
  ret i32 %29
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"Debian clang version 19.1.7 (7)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
