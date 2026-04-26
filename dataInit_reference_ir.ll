%struct.struct0 = type { i32, i32, i32, i32, i32, i32 }
define dso_local void @dataInit(%struct.struct0* %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6) {
%8 = alloca %struct.struct0*, align 8
%9 = alloca i32, align 4
%10 = alloca i32, align 4
%11 = alloca i32, align 4
%12 = alloca i32, align 4
%13 = alloca i32, align 4
%14 = alloca i32, align 4
store %struct.struct0* %0, %struct.struct0** %8, align 8
store i32 %1, i32* %9, align 4
store i32 %2, i32* %10, align 4
store i32 %3, i32* %11, align 4
store i32 %4, i32* %12, align 4
store i32 %5, i32* %13, align 4
store i32 %6, i32* %14, align 4
%15 = load i32, i32* %9, align 4
%16 = load %struct.struct0*, %struct.struct0** %8, align 8
%17 = getelementptr inbounds %struct.struct0, %struct.struct0* %16, i32 0, i32 0
store i32 %15, i32* %17, align 4
%18 = load i32, i32* %10, align 4
%19 = load %struct.struct0*, %struct.struct0** %8, align 8
%20 = getelementptr inbounds %struct.struct0, %struct.struct0* %19, i32 0, i32 1
store i32 %18, i32* %20, align 4
%21 = load i32, i32* %11, align 4
%22 = load %struct.struct0*, %struct.struct0** %8, align 8
%23 = getelementptr inbounds %struct.struct0, %struct.struct0* %22, i32 0, i32 2
store i32 %21, i32* %23, align 4
%24 = load i32, i32* %12, align 4
%25 = load %struct.struct0*, %struct.struct0** %8, align 8
%26 = getelementptr inbounds %struct.struct0, %struct.struct0* %25, i32 0, i32 3
store i32 %24, i32* %26, align 4
%27 = load i32, i32* %13, align 4
%28 = load %struct.struct0*, %struct.struct0** %8, align 8
%29 = getelementptr inbounds %struct.struct0, %struct.struct0* %28, i32 0, i32 4
store i32 %27, i32* %29, align 4
%30 = load i32, i32* %14, align 4
%31 = load %struct.struct0*, %struct.struct0** %8, align 8
%32 = getelementptr inbounds %struct.struct0, %struct.struct0* %31, i32 0, i32 5
store i32 %30, i32* %32, align 4
ret void
}
