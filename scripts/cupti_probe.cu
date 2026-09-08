// CUPTI Range Profiler probe: the acceptance test for Tier-4 hardware counters
// on the CUDA backend. Runs the full capture sequence against a saxpy kernel and
// prints per-range metric values.
//
//   nvcc -arch=sm_86 scripts/cupti_probe.cu -o /tmp/cupti_probe \
//        -I/opt/cuda/include -L/opt/cuda/lib64 -lcupti -lcuda
//   LD_LIBRARY_PATH=/opt/cuda/lib64 /tmp/cupti_probe
//
// Counter collection is admin-gated by default (`RmProfilingAdminOnly: 1` in
// /proc/driver/nvidia/params). Unprivileged, everything up to and including
// config-image construction succeeds and cuptiRangeProfilerStart fails with
// CUPTI_ERROR_INSUFFICIENT_PRIVILEGES (35). To lift it:
//
//   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
//     | sudo tee /etc/modprobe.d/nvidia-profiling.conf
//   sudo mkinitcpio -P && reboot
//
// Running the probe under sudo confirms the same thing without a reboot.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cupti_profiler_host.h>
#include <cupti_range_profiler.h>

#define CK(x) do{ CUptiResult _r=(x); if(_r!=CUPTI_SUCCESS){ const char*s="?"; \
  cuptiGetResultString(_r,&s); printf("FAIL %s -> %d (%s)\n", #x, (int)_r, s); exit(1);} }while(0)

__global__ void saxpy(float* y, const float* x, float a, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

static const char* kMetrics[] = {
  "sm__cycles_active.sum",
  "sm__warps_launched.sum",
  "smsp__inst_executed.sum",
  "dram__bytes.sum",
};
static const size_t kNum = sizeof(kMetrics)/sizeof(kMetrics[0]);

int main(void){
  CUdevice dev; CUcontext ctx;
  cuInit(0); cuDeviceGet(&dev,0); cuDevicePrimaryCtxRetain(&ctx,dev); cuCtxSetCurrent(ctx);

  CUpti_Profiler_Initialize_Params ip={CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CK(cuptiProfilerInitialize(&ip));

  CUpti_Device_GetChipName_Params cn={CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  cn.deviceIndex=0; CK(cuptiDeviceGetChipName(&cn));
  printf("chip: %s\n", cn.pChipName);

  CUpti_Profiler_GetCounterAvailability_Params ca={CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
  ca.ctx=ctx; CK(cuptiProfilerGetCounterAvailability(&ca));
  uint8_t* avail=(uint8_t*)malloc(ca.counterAvailabilityImageSize);
  ca.pCounterAvailabilityImage=avail;
  { CUptiResult r=cuptiProfilerGetCounterAvailability(&ca); const char*m="?"; cuptiGetResultString(r,&m);
    printf("CA fetch-image -> %d (%s)%s\n",(int)r,m, r==CUPTI_SUCCESS?"":"  [falling back to NULL image]");
    if(r!=CUPTI_SUCCESS){ free(avail); avail=NULL; } }

  CUpti_Profiler_Host_Initialize_Params hi={CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
  hi.profilerType=CUPTI_PROFILER_TYPE_RANGE_PROFILER; hi.pChipName=cn.pChipName;
  hi.pCounterAvailabilityImage=avail; CK(cuptiProfilerHostInitialize(&hi));
  printf("host init OK (availability image %s)\n", avail?"real":"NULL");

  CUpti_Profiler_Host_ConfigAddMetrics_Params am={CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
  am.pHostObject=hi.pHostObject; am.ppMetricNames=kMetrics; am.numMetrics=kNum;
  CK(cuptiProfilerHostConfigAddMetrics(&am));

  CUpti_Profiler_Host_GetConfigImageSize_Params cs={CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
  cs.pHostObject=hi.pHostObject; CK(cuptiProfilerHostGetConfigImageSize(&cs));
  uint8_t* cfg=(uint8_t*)malloc(cs.configImageSize);
  CUpti_Profiler_Host_GetConfigImage_Params ci={CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
  ci.pHostObject=hi.pHostObject; ci.configImageSize=cs.configImageSize; ci.pConfigImage=cfg;
  CK(cuptiProfilerHostGetConfigImage(&ci));

  CUpti_Profiler_Host_GetNumOfPasses_Params np={CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
  np.configImageSize=cs.configImageSize; np.pConfigImage=cfg;
  CK(cuptiProfilerHostGetNumOfPasses(&np));
  printf("config: %zu bytes, %zu pass(es)\n", cs.configImageSize, np.numOfPasses);

  CUpti_RangeProfiler_Enable_Params en={CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
  en.ctx=ctx; CK(cuptiRangeProfilerEnable(&en));

  CUpti_RangeProfiler_GetCounterDataSize_Params ds={CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
  ds.pRangeProfilerObject=en.pRangeProfilerObject; ds.pMetricNames=kMetrics; ds.numMetrics=kNum;
  ds.maxNumOfRanges=16; ds.maxNumRangeTreeNodes=16; CK(cuptiRangeProfilerGetCounterDataSize(&ds));
  uint8_t* cd=(uint8_t*)malloc(ds.counterDataSize);
  CUpti_RangeProfiler_CounterDataImage_Initialize_Params cdi={CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  cdi.pRangeProfilerObject=en.pRangeProfilerObject; cdi.counterDataSize=ds.counterDataSize; cdi.pCounterData=cd;
  CK(cuptiRangeProfilerCounterDataImageInitialize(&cdi));

  CUpti_RangeProfiler_SetConfig_Params sc={CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
  sc.pRangeProfilerObject=en.pRangeProfilerObject;
  sc.configSize=cs.configImageSize; sc.pConfig=cfg;
  sc.counterDataImageSize=ds.counterDataSize; sc.pCounterDataImage=cd;
  sc.range=CUPTI_AutoRange; sc.replayMode=CUPTI_KernelReplay;
  sc.maxRangesPerPass=16; sc.numNestingLevels=1; sc.minNestingLevel=1;
  sc.passIndex=0; sc.targetNestingLevel=1;
  CK(cuptiRangeProfilerSetConfig(&sc));

  int n=1<<20; float *x,*y; cudaMalloc(&x,n*4); cudaMalloc(&y,n*4);
  cudaMemset(x,0,n*4); cudaMemset(y,0,n*4);

  CUpti_RangeProfiler_Start_Params st={CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
  st.pRangeProfilerObject=en.pRangeProfilerObject; CK(cuptiRangeProfilerStart(&st));

  saxpy<<<(n+255)/256,256>>>(y,x,2.0f,n);
  cudaDeviceSynchronize();

  CUpti_RangeProfiler_Stop_Params sp={CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
  sp.pRangeProfilerObject=en.pRangeProfilerObject; CK(cuptiRangeProfilerStop(&sp));
  printf("allPassSubmitted=%u passIndex=%zu\n", sp.isAllPassSubmitted, sp.passIndex);

  CUpti_RangeProfiler_DecodeData_Params dd={CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
  dd.pRangeProfilerObject=en.pRangeProfilerObject; CK(cuptiRangeProfilerDecodeData(&dd));
  printf("rangesDropped=%zu\n", dd.numOfRangeDropped);

  CUpti_RangeProfiler_GetCounterDataInfo_Params di={CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE};
  di.pCounterDataImage=cd; di.counterDataImageSize=ds.counterDataSize;
  CK(cuptiRangeProfilerGetCounterDataInfo(&di));
  printf("numTotalRanges=%zu\n", di.numTotalRanges);

  double* vals=(double*)malloc(kNum*sizeof(double));
  for(size_t r=0;r<di.numTotalRanges;r++){
    CUpti_RangeProfiler_CounterData_GetRangeInfo_Params ri={CUpti_RangeProfiler_CounterData_GetRangeInfo_Params_STRUCT_SIZE};
    ri.pCounterDataImage=cd; ri.counterDataImageSize=ds.counterDataSize; ri.rangeIndex=r; ri.rangeDelimiter="/";
    CK(cuptiRangeProfilerCounterDataGetRangeInfo(&ri));
    CUpti_Profiler_Host_EvaluateToGpuValues_Params ev={CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    ev.pHostObject=hi.pHostObject; ev.pCounterDataImage=cd; ev.counterDataImageSize=ds.counterDataSize;
    ev.rangeIndex=r; ev.ppMetricNames=kMetrics; ev.numMetrics=kNum; ev.pMetricValues=vals;
    CK(cuptiProfilerHostEvaluateToGpuValues(&ev));
    printf("range[%zu] \"%s\"\n", r, ri.rangeName);
    for(size_t m=0;m<kNum;m++) printf("    %-32s %.0f\n", kMetrics[m], vals[m]);
  }

  CUpti_RangeProfiler_Disable_Params dis={CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
  dis.pRangeProfilerObject=en.pRangeProfilerObject; CK(cuptiRangeProfilerDisable(&dis));
  printf("\n== counters captured OK ==\n");
  return 0;
}
