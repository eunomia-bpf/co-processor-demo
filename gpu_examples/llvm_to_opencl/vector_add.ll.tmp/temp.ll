; ModuleID = 'vector_add.ll'
source_filename = "vector_add.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@opencl.ocl.version = constant [3 x i32] [i32 2, i32 0, i32 0]

define spir_kernel void @vector_add(float addrspace(1)* %a, float addrspace(1)* %b, float addrspace(1)* %c, i32 %n) {
entry:
  %gid = call i32 @get_global_id(i32 0)
  %cmp = icmp ult i32 %gid, %n
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %idx = zext i32 %gid to i64
  %a_ptr = getelementptr inbounds float, float addrspace(1)* %a, i64 %idx
  %a_val = load float, float addrspace(1)* %a_ptr, align 4
  %b_ptr = getelementptr inbounds float, float addrspace(1)* %b, i64 %idx
  %b_val = load float, float addrspace(1)* %b_ptr, align 4
  %sum = fadd float %a_val, %b_val
  %c_ptr = getelementptr inbounds float, float addrspace(1)* %c, i64 %idx
  store float %sum, float addrspace(1)* %c_ptr, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare i32 @get_global_id(i32)

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
