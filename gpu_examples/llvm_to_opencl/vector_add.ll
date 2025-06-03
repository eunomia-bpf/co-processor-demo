; LLVM IR for vector addition kernel
; This is a conceptual representation of what the vector_add.cl kernel would look like in LLVM IR
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; OpenCL kernel attributes
@opencl.ocl.version = constant [3 x i32] [i32 2, i32 0, i32 0]

; Vector addition kernel function
define spir_kernel void @vector_add(float addrspace(1)* %a, float addrspace(1)* %b, float addrspace(1)* %c, i32 %n) {
entry:
  ; Get global ID
  %gid = call i32 @get_global_id(i32 0)
  
  ; Check if global ID is within bounds
  %cmp = icmp ult i32 %gid, %n
  br i1 %cmp, label %if.then, label %if.end

if.then:
  ; Load a[gid]
  %idx = zext i32 %gid to i64
  %a_ptr = getelementptr inbounds float, float addrspace(1)* %a, i64 %idx
  %a_val = load float, float addrspace(1)* %a_ptr, align 4
  
  ; Load b[gid]
  %b_ptr = getelementptr inbounds float, float addrspace(1)* %b, i64 %idx
  %b_val = load float, float addrspace(1)* %b_ptr, align 4
  
  ; Compute a[gid] + b[gid]
  %sum = fadd float %a_val, %b_val
  
  ; Store result in c[gid]
  %c_ptr = getelementptr inbounds float, float addrspace(1)* %c, i64 %idx
  store float %sum, float addrspace(1)* %c_ptr, align 4
  br label %if.end

if.end:
  ret void
}

; OpenCL kernel metadata
!opencl.kernels = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!1}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @vector_add}
!1 = !{i32 2, i32 0}
!2 = !{}

; OpenCL built-in function declaration
declare i32 @get_global_id(i32) 