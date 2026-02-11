@.str = external hidden unnamed_addr constant [2 x i32], align 4
define dso_local void @i64() local_unnamed_addr {
br label %1
1: ; preds = %4, %0
%2 = phi i64 [ %6, %4 ], [ 0, %0 ]
%3 = icmp eq i64 %2, 16
br i1 %3, label %7, label %4
4: ; preds = %1
%5 = getelementptr inbounds [2 x i32], [2 x i32]* @.str, i64 0, i64 %2
store i32 0, i32* %5, align 4
%6 = add nuw nsw i64 %2, 1
br label %1
7: ; preds = %1
ret void
}