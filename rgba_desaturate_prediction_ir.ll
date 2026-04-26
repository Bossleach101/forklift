%struct.struct0 = type { i32, i32, i32, i32 }
define dso_local void @rgba_desaturate(%struct.struct0* %0, %struct.struct0* %1) {
%3 = alloca %struct.struct0*, align 8
%4 = alloca %struct.struct0*, align 8
%5 = alloca double, align 8
store %struct.struct0* %0, %struct.struct0** %3, align 8
store %struct.struct0* %1, %struct.struct0** %4, align 8
%6 = load %struct.struct0*, %struct.struct0** %4, align 8
%7 = getelementptr inbounds %struct.struct0, %struct.struct0* %6, i32 0, i32 0
%8 = load i32, i32* %7, align 4
%9 = load %struct.struct0*, %struct.struct0** %4, align 8
%10 = getelementptr inbounds %struct.struct0, %struct.struct0* %9, i32 0, i32 1
%11 = load i32, i32* %10, align 4
%12 = add nsw i32 %8, %11
%13 = load %struct.struct0*, %struct.struct0** %4, align 8
%14 = getelementptr inbounds %struct.struct0, %struct.struct0* %13, i32 0, i32 2
%15 = load i32, i32* %14, align 4
%16 = add nsw i32 %12, %15
%17 = sdiv i32 %16, 3
%18 = sitofp i32 %17 to double
store double %18, double* %5, align 8
%19 = load double, double* %5, align 8
%20 = fptosi double %19 to i32
%21 = load %struct.struct0*, %struct.struct0** %3, align 8
%22 = getelementptr inbounds %struct.struct0, %struct.struct0* %21, i32 0, i32 0
store i32 %20, i32* %22, align 4
%23 = load double, double* %5, align 8
%24 = fptosi double %23 to i32
%25 = load %struct.struct0*, %struct.struct0** %3, align 8
%26 = getelementptr inbounds %struct.struct0, %struct.struct0* %25, i32 0, i32 1
store i32 %24, i32* %26, align 4
%27 = load double, double* %5, align 8
%28 = fptosi double %27 to i32
%29 = load %struct.struct0*, %struct.struct0** %3, align 8
%30 = getelementptr inbounds %struct.struct0, %struct.struct0* %29, i32 0, i32 2
store i32 %28, i32* %30, align 4
%31 = load %struct.struct0*, %struct.struct0** %4, align 8
%32 = getelementptr inbounds %struct.struct0, %struct.struct0* %31, i32 0, i32 3
%33 = load i32, i32* %32, align 4
%34 = load %struct.struct0*, %struct.struct0** %3, align 8
%35 = getelementptr inbounds %struct.struct0, %struct.struct0* %34, i32 0, i32 3
store i32 %33, i32* %35, align 4
ret void
}