# 1 "gemm.cu"
# 151 "/usr/include/stdio.h" 3
extern FILE *stderr;
# 33 "../../common/polybench.c"
extern int polybench_papi_counters_threadid;
extern double polybench_program_total_flops;
# 51 "../../common/polybench.c"
double polybench_t_start = 0;
# 51 "../../common/polybench.c"
double polybench_t_end = 0;

unsigned long long polybench_c_start = 0;
# 53 "../../common/polybench.c"
unsigned long long polybench_c_end = 0;
static const char __T0[29];
# 33 "../../common/polybench.c"
int polybench_papi_counters_threadid = 0;
double polybench_program_total_flops = (0.0);
static const char __T0[29] = "void polybench_flush_cache()";
