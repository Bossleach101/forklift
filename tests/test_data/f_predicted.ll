%struct.struct0 = type { i32 }
define dso_local void @f(%struct.struct0* nocapture %0, i32 %1, i32 %2) local_unnamed_addr {
%4 = sext i32 %2 to i64
br label %5
5: ; preds = %8, %3
%6 = phi i64 [ %12, %8 ], [ 0, %3 ]
%7 = icmp slt i64 %6, %4
br i1 %7, label %8, label %13
8: ; preds = %5
%9 = getelementptr inbounds %struct.struct0, %struct.struct0* %0, i64 %6, i32 0
%10 = load i32, i32* %9, align 4
%11 = add nsw i32 %10, %1
store i32 %11, i32* %9, align 4
%12 = add nuw nsw i64 %6, 1
br label %5
13: ; preds = %5
ret void
}
