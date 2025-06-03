; Simple LLVM IR for vector addition
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; OpenCL kernel attributes
@opencl.ocl.version = constant [3 x i32] [i32 2, i32 0, i32 0]

; Vector addition kernel function
define spir_kernel void @vector_add(float* %a, float* %b, float* %c) {
  ; Get global ID
  %gid = call i32 @get_global_id(i32 0)
  
  ; Load a[gid]
  %idx = zext i32 %gid to i64
  %a_ptr = getelementptr inbounds float, float* %a, i64 %idx
  %a_val = load float, float* %a_ptr, align 4
  
  ; Load b[gid]
  %b_ptr = getelementptr inbounds float, float* %b, i64 %idx
  %b_val = load float, float* %b_ptr, align 4
  
  ; Compute a[gid] + b[gid]
  %sum = fadd float %a_val, %b_val
  
  ; Store result in c[gid]
  %c_ptr = getelementptr inbounds float, float* %c, i64 %idx
  store float %sum, float* %c_ptr, align 4
  
  ret void
}

; OpenCL built-in function declaration
declare i32 @get_global_id(i32) 