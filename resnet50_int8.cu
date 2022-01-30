#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
#include <sm_61_intrinsics.h>
#endif
#include <mma.h>
extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__11_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[16];
  __shared__ int pad_data_shared[880];
  __shared__ int placeholder_shared[512];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 8; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) < 880) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.z)) < 63) {
            ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 1605632) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) / 440) * 802816)) + (ic_chunk_outer * 100352)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) % 440) / 55) * 12544)) + (((int)blockIdx.x) * 448)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) % 55) * 4)))))[0];
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.z) * 14)) + ((int)threadIdx.x)) < 512) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.z)) < 37) {
          if ((((((int)blockIdx.y) * 16) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 7)) + (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) >> 5)) < 32) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 16384) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 7168)) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) >> 5) * 1024)) + (ic_chunk_outer * 128)) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) & 31) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 8; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 220) + (((int)threadIdx.x) * 8)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 220) + (((int)threadIdx.x) * 8)) + 1760))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 220) + (((int)threadIdx.x) * 8)) + 112))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 220) + (((int)threadIdx.x) * 8)) + 1872))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 12))]);
      }
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[(((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147466095) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147466095) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1320312794) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 100352))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147466095) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147466095) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1320312794) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 56))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147466095) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147466095) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1320312794) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 100408))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147466095) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1255108819) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147466095) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1320312794) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__9_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[28];
  __shared__ int pad_data_shared[896];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 12))] = 0;
    compute[((oc_block_init + 16))] = 0;
    compute[((oc_block_init + 20))] = 0;
    compute[((oc_block_init + 24))] = 0;
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 8; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 401408) + (ic_chunk_outer * 50176)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 56) * 3136)) + (((int)blockIdx.x) * 224)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 56) * 4)))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 32768) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 4096)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 4) * 2048)) + (ic_chunk_outer * 256)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 15) * 16)) + (((int)threadIdx.x) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 16))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 32))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 48))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 12))]);
        compute[((oc_block + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 64))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 16))]);
        compute[((oc_block + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 80))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 20))]);
        compute[((oc_block + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 96))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 24))]);
      }
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1095017033) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 16))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1095017033) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 32))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1095017033) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 48))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1095017033) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 64))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1095017033) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 80))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1095017033) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 96))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1442894291) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147471922) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1095017033) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__6_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[16];
  __shared__ int placeholder_shared[2048];
  __shared__ int pad_data_shared[896];
  #pragma unroll
  for (int n_init = 0; n_init < 2; ++n_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((n_init * 4) + oc_block_init))] = 0;
      compute[((((n_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 37) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((int)blockIdx.y) * 65536) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 4096)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 401408) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 200704)) + (ic_chunk_outer_outer * 12544)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 37) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.y) * 65536) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 4096)) + (ic_chunk_outer_outer * 256)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((n * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n * 1792) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((n * 4) + oc_block))]);
          compute[((((n * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n * 1792) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((n * 4) + oc_block) + 8))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 401408) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) >> 4) * 200704)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + 188160))))[0];
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int n1 = 0; n1 < 2; ++n1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((n1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 1792) + (ic_chunk_inner1 * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(((n1 * 4) + oc_block1))]);
        compute[((((n1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 1792) + (ic_chunk_inner1 * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((((n1 * 4) + oc_block1) + 8))]);
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 2; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1250824417) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1250824417) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147474446) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1250824417) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1250824417) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147474446) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1228982520) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1250824417) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1250824417) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147474446) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1250824417) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1250824417) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147474446) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0)))) * (long)1228982520) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_cast_add_right_shift_clip_cast_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 1605632) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)min((int)(((((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] + 8388608) >> 24)), (int)(127)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_15119380522063600768__kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[28];
  __shared__ int pad_data_shared[1568];
  __shared__ int placeholder_shared[2048];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((int)blockIdx.z) * 25088) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0];
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 10; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 1024) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.y)) < 147) {
          ((int*)((signed char*)placeholder_shared + ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 6) * 256) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 15) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 32768) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 6) * 2048)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 15) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0];
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 7; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 3136) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 25088) + (ic_chunk_outer_outer * 3136)) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)) + 3136))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 10; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + ((int)threadIdx.y)) < 147) {
            ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 6) * 256)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 15) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 32768) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 6) * 2048)) + (ic_chunk_outer_outer * 256)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 15) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)) + 256))))[0];
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oh = 0; oh < 7; ++oh) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3136) + (ic_chunk_inner * 196)) + (oh * 28)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oh1 = 0; oh1 < 7; ++oh1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner1 * 196) + (oh1 * 28)) + (((int)threadIdx.x) * 4)) + 3136))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(((oh1 * 4) + oc_block1))]);
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[(((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (ax2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1353888287) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1353888287) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147483342) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1353888287) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1353888287) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147483342) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1840126696) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (ax2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (ax2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)2056266911) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_2_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 1605632) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)2131420559) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__11_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[16];
  __shared__ int pad_data_shared[288];
  __shared__ int placeholder_shared[2304];
  #pragma unroll
  for (int oh_init = 0; oh_init < 2; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
      compute[((((oh_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 144) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 36) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 3) {
            ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6))) && ((((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6)) < 29)) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)))) && ((((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)) < 29)) ? (int)((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 72) * 100352)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 72) / 36) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6) * 4)) - 116))))[0] : (int)(int)0);
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 18) * 4608)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)))))[0];
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 144) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 36) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 3) {
              ((int*)((signed char*)pad_data_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 576) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6))) && ((((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6)) < 29)) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)))) && ((((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)) < 29)) ? (int)((int*)((signed char*)placeholder + ((((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 72) * 100352)) + (ic_chunk_outer_outer * 6272)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 72) / 36) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6) * 4)) + 6156))))[0] : (int)(int)0);
          }
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 4608) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 18) * 4608)) + (ic_chunk_outer_outer * 288)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)) + 288))))[0];
    }
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
        #pragma unroll
        for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
          #pragma unroll
          for (int oh = 0; oh < 2; ++oh) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 576) + (((int)threadIdx.z) * 288)) + (ic_chunk_inner * 144)) + (oh * 24)) + (kh_inner * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 4608) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
              compute[((((oh * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((ic_chunk_outer_outer & 1) * 576) + (((int)threadIdx.z) * 288)) + (ic_chunk_inner * 144)) + (oh * 24)) + (kh_inner * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 48))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 4608) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((((oh * 4) + oc_block) + 8))]);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 2; ++ic_chunk_inner1) {
      #pragma unroll
      for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
        #pragma unroll
        for (int oh1 = 0; oh1 < 2; ++oh1) {
          #pragma unroll
          for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
            compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 288) + (ic_chunk_inner1 * 144)) + (oh1 * 24)) + (kh_inner1 * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 576))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 4608))))[0], compute[(((oh1 * 4) + oc_block1))]);
            compute[((((oh1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 288) + (ic_chunk_inner1 * 144)) + (oh1 * 24)) + (kh_inner1 * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 624))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 4608))))[0], compute[((((oh1 * 4) + oc_block1) + 8))]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 100352)) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1818067853) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1818067853) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1398390528) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      ((signed char*)T_cast)[(((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 100352)) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))]) << ((long)16)) : ((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))])) * (long)1818067853) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))]) << ((long)16)) : ((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))])) * (long)1818067853) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1398390528) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_12402219635377536017__1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[16];
  __shared__ int pad_data_shared[1792];
  __shared__ int placeholder_shared[2048];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 50176)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)))))[0];
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 37) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 16384) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 3; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 3584) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) >> 4) * 50176)) + (ic_chunk_outer_outer * 12544)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + 12544))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 37) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 16384) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (ic_chunk_outer_outer * 256)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)) + 1792))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)) + 1792))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 12))]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
      compute[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 3584))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(oc_block1)]);
      compute[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 5376))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[((oc_block1 + 8))]);
      compute[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 3584))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((oc_block1 + 4))]);
      compute[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 5376))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((oc_block1 + 12))]);
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((int*)T_relu)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073757977) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073757977) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1719164685) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder4)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))])), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073757977) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073757977) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1719164685) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))])), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1073757977) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1073757977) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1719164685) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))])), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1073757977) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)2105581540) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1073757977) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1719164685) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))])), (int)(0));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_12402219635377536017__3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[16];
  __shared__ int pad_data_shared[128];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oc_chunk_init * 4) + oc_block_init))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ic_chunk_outer = 0; ic_chunk_outer < 2; ++ic_chunk_outer) {
    __syncthreads();
      ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ic_chunk_outer * 100352)) + ((((int)threadIdx.y) >> 1) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.y) & 1) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)))))[0];
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 8192) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 1024)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 2)) >> 3) * 256)) + (ic_chunk_outer * 128)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 2)) & 7) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 8; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((oc_chunk * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (oc_chunk * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((oc_chunk * 4) + oc_block))]);
          compute[((((oc_chunk * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)) + 32))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (oc_chunk * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((((oc_chunk * 4) + oc_block) + 8))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1439495920) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1439495920) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147247346) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1439495920) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1439495920) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147247346) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1919367107) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder4)[(((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))])), (int)(0));
      ((int*)T_relu)[((((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1439495920) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1439495920) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147247346) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1439495920) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1439495920) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147247346) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1919367107) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder4)[((((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 224))])), (int)(0));
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_4_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 1605632) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1978965331) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ compute, void* __restrict__ placeholder2) {
  int compute1[112];
  __shared__ int pad_data_shared[104];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int n_init = 0; n_init < 2; ++n_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute1[(((n_init * 4) + oc_block_init))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 56))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 8))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 64))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 16))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 72))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 24))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 80))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 32))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 88))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 40))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 96))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 48))] = 0;
      compute1[((((n_init * 4) + oc_block_init) + 104))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + ((int)threadIdx.z)) < 52) {
        ((int*)((signed char*)pad_data_shared + (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 401408) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + ((int)threadIdx.z)) / 26) * 200704)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + ((int)threadIdx.z)) % 26) / 13) * 784)) + (((int)blockIdx.x) * 112)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + ((int)threadIdx.z)) % 13) * 4)))))[0];
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 16; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)placeholder_shared + (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 262144) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16384)) + ((((int)threadIdx.z) >> 3) * 4096)) + ((((int)threadIdx.z) & 7) * 4)))))[0];
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 127; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 32) + ((int)threadIdx.z)) < 52) {
          ((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer + 1) & 1) * 208) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128)) + (((int)threadIdx.z) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 401408) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 32) + ((int)threadIdx.z)) / 26) * 200704)) + (ic_chunk_outer_outer * 1568)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 32) + ((int)threadIdx.z)) % 26) / 13) * 784)) + (((int)blockIdx.x) * 112)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 32) + ((int)threadIdx.z)) % 13) * 4)) + 1568))))[0];
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 16; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer + 1) & 1) * 2048) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128)) + (((int)threadIdx.z) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 262144) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16384)) + ((((int)threadIdx.z) >> 3) * 4096)) + (ic_chunk_outer_outer * 32)) + ((((int)threadIdx.z) & 7) * 4)) + 32))))[0];
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
      #pragma unroll
      for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute1[(((n * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[(((n * 4) + oc_block))]);
          compute1[((((n * 4) + oc_block) + 56))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 1024))))[0], compute1[((((n * 4) + oc_block) + 56))]);
          compute1[((((n * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)) + 8))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((((n * 4) + oc_block) + 8))]);
          compute1[((((n * 4) + oc_block) + 64))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)) + 8))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 1024))))[0], compute1[((((n * 4) + oc_block) + 64))]);
          compute1[((((n * 4) + oc_block) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)) + 16))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((((n * 4) + oc_block) + 16))]);
          compute1[((((n * 4) + oc_block) + 72))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)) + 16))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 1024))))[0], compute1[((((n * 4) + oc_block) + 72))]);
          compute1[((((n * 4) + oc_block) + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)) + 24))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((((n * 4) + oc_block) + 24))]);
          compute1[((((n * 4) + oc_block) + 80))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)) + 24))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 1024))))[0], compute1[((((n * 4) + oc_block) + 80))]);
          compute1[((((n * 4) + oc_block) + 32))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)) + 32))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((((n * 4) + oc_block) + 32))]);
          compute1[((((n * 4) + oc_block) + 88))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)) + 32))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 1024))))[0], compute1[((((n * 4) + oc_block) + 88))]);
          compute1[((((n * 4) + oc_block) + 40))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)) + 40))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((((n * 4) + oc_block) + 40))]);
          compute1[((((n * 4) + oc_block) + 96))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)) + 40))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 1024))))[0], compute1[((((n * 4) + oc_block) + 96))]);
          compute1[((((n * 4) + oc_block) + 48))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)) + 48))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((((n * 4) + oc_block) + 48))]);
          compute1[((((n * 4) + oc_block) + 104))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 208) + (n * 104)) + (ic_chunk_inner * 52)) + 48))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.z) * 32)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 1024))))[0], compute1[((((n * 4) + oc_block) + 104))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 2; ++ic_chunk_inner1) {
    #pragma unroll
    for (int n1 = 0; n1 < 2; ++n1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute1[(((n1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 208))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute1[(((n1 * 4) + oc_block1))]);
        compute1[((((n1 * 4) + oc_block1) + 56))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 208))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 3072))))[0], compute1[((((n1 * 4) + oc_block1) + 56))]);
        compute1[((((n1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 216))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute1[((((n1 * 4) + oc_block1) + 8))]);
        compute1[((((n1 * 4) + oc_block1) + 64))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 216))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 3072))))[0], compute1[((((n1 * 4) + oc_block1) + 64))]);
        compute1[((((n1 * 4) + oc_block1) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 224))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute1[((((n1 * 4) + oc_block1) + 16))]);
        compute1[((((n1 * 4) + oc_block1) + 72))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 224))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 3072))))[0], compute1[((((n1 * 4) + oc_block1) + 72))]);
        compute1[((((n1 * 4) + oc_block1) + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 232))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute1[((((n1 * 4) + oc_block1) + 24))]);
        compute1[((((n1 * 4) + oc_block1) + 80))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 232))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 3072))))[0], compute1[((((n1 * 4) + oc_block1) + 80))]);
        compute1[((((n1 * 4) + oc_block1) + 32))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 240))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute1[((((n1 * 4) + oc_block1) + 32))]);
        compute1[((((n1 * 4) + oc_block1) + 88))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 240))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 3072))))[0], compute1[((((n1 * 4) + oc_block1) + 88))]);
        compute1[((((n1 * 4) + oc_block1) + 40))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 248))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute1[((((n1 * 4) + oc_block1) + 40))]);
        compute1[((((n1 * 4) + oc_block1) + 96))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 248))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 3072))))[0], compute1[((((n1 * 4) + oc_block1) + 96))]);
        compute1[((((n1 * 4) + oc_block1) + 48))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 256))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute1[((((n1 * 4) + oc_block1) + 48))]);
        compute1[((((n1 * 4) + oc_block1) + 104))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 104) + (ic_chunk_inner1 * 52)) + 256))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 32) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 3072))))[0], compute1[((((n1 * 4) + oc_block1) + 104))]);
      }
    }
  }
  #pragma unroll
  for (int i0_inner_inner_inner_inner = 0; i0_inner_inner_inner_inner < 2; ++i0_inner_inner_inner_inner) {
    #pragma unroll
    for (int i4 = 0; i4 < 4; ++i4) {
      ((int*)compute)[(((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[(((i0_inner_inner_inner_inner * 4) + i4))]) << ((long)18)) : ((long)compute1[(((i0_inner_inner_inner_inner * 4) + i4))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[(((i0_inner_inner_inner_inner * 4) + i4))]) << ((long)18)) : ((long)compute1[(((i0_inner_inner_inner_inner * 4) + i4))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 6272))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 56))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 56))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 56))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 56))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 4))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 8))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 8))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 8))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 8))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 6276))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 64))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 64))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 64))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 64))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 8))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 16))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 16))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 16))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 16))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 6280))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 72))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 72))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 72))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 72))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 12))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 24))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 24))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 24))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 24))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 6284))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 80))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 80))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 80))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 80))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 16))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 32))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 32))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 32))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 32))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 6288))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 88))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 88))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 88))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 88))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 20))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 40))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 40))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 40))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 40))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 6292))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 96))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 96))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 96))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 96))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 24))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 48))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 48))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 48))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 48))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (i0_inner_inner_inner_inner * 100352)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.x) * 28)) + i4) + 6296))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 104))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 104))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 104))]) << ((long)18)) : ((long)compute1[((((i0_inner_inner_inner_inner * 4) + i4) + 104))])) * (long)1168676628) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 256) + (((int)threadIdx.z) * 4)) + i4) + 128))]))) * (long)1145619329) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__5_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[28];
  __shared__ int pad_data_shared[256];
  __shared__ int placeholder_shared[576];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
    ((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= ((int)threadIdx.y)) && (((int)threadIdx.y) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) - 60))))[0] : (int)(int)0);
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 63; ++ic_chunk_outer_outer) {
    __syncthreads();
      ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= ((int)threadIdx.y)) && (((int)threadIdx.y) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (ic_chunk_outer_outer * 784)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) + 724))))[0] : (int)(int)0);
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 144) {
        if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 9) {
              ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 9) * 9216)) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 7; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 512) + (((int)threadIdx.z) * 256)) + ((((int)threadIdx.x) >> 1) * 112)) + (oh * 16)) + (kh_inner * 16)) + (kw_inner * 4)) + ((((int)threadIdx.x) & 1) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 144) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 9) {
            ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 9) * 9216)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)) + 9072))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 7; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 256) + ((((int)threadIdx.x) >> 1) * 112)) + (oh1 * 16)) + (kh_inner1 * 16)) + (kw_inner1 * 4)) + ((((int)threadIdx.x) & 1) * 4)) + 512))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + ((((int)threadIdx.x) >> 1) * 392)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((17 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1266188500) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((17 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1266188500) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1185323653) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_8_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 49; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1769513466) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_15119380522063600768__7_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[32];
  __shared__ int pad_data_shared[3584];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oc_chunk_init * 4) + oc_block_init))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 16))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 8))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 24))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 100352) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 6272)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) >> 2) * 3136)) + (((int)blockIdx.x) * 448)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) & 3) * 112)) + ((((int)threadIdx.x) % 28) * 4)))))[0];
  }
  #pragma unroll
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 1; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)) + 7168))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 6272)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) >> 2) * 3136)) + (((int)blockIdx.x) * 448)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) & 3) * 112)) + ((((int)threadIdx.x) % 28) * 4)) + 50176))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 4) + ((int)threadIdx.y)) < 19) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 8192) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) >> 4) * 512)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((oc_chunk * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((oc_chunk * 4) + oc_block))]);
          compute[((((oc_chunk * 4) + oc_block) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((oc_chunk * 4) + oc_block) + 16))]);
          compute[((((oc_chunk * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)) + 224))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((((oc_chunk * 4) + oc_block) + 8))]);
          compute[((((oc_chunk * 4) + oc_block) + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)) + 224))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((oc_chunk * 4) + oc_block) + 24))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 4) + ((int)threadIdx.y)) < 19) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 8192) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) >> 4) * 512)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_chunk1 = 0; oc_chunk1 < 2; ++oc_chunk1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((oc_chunk1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7168))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oc_chunk1 * 4) + oc_block1))]);
        compute[((((oc_chunk1 * 4) + oc_block1) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7168))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute[((((oc_chunk1 * 4) + oc_block1) + 16))]);
        compute[((((oc_chunk1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7392))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((((oc_chunk1 * 4) + oc_block1) + 8))]);
        compute[((((oc_chunk1 * 4) + oc_block1) + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7392))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute[((((oc_chunk1 * 4) + oc_block1) + 24))]);
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147416796) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147416796) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1467685356) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)2136391595) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2147416796) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2147416796) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)1467685356) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))])) * (long)2136391595) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147416796) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147416796) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1467685356) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 224))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 224))])) * (long)2136391595) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25312))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2147416796) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1268844533) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2147416796) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)1467685356) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25312))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25312))])) * (long)2136391595) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[16];
  __shared__ int placeholder_shared[2048];
  __shared__ int pad_data_shared[896];
  #pragma unroll
  for (int n_init = 0; n_init < 2; ++n_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((n_init * 4) + oc_block_init))] = 0;
      compute[((((n_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 37) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((int)blockIdx.y) * 65536) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 4096)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 401408) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 200704)) + (ic_chunk_outer_outer * 12544)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 37) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.y) * 65536) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 4096)) + (ic_chunk_outer_outer * 256)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((n * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n * 1792) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((n * 4) + oc_block))]);
          compute[((((n * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n * 1792) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((n * 4) + oc_block) + 8))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 401408) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) >> 4) * 200704)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + 188160))))[0];
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int n1 = 0; n1 < 2; ++n1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((n1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 1792) + (ic_chunk_inner1 * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(((n1 * 4) + oc_block1))]);
        compute[((((n1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 1792) + (ic_chunk_inner1 * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((((n1 * 4) + oc_block1) + 8))]);
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 2; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)2032905037) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)2032905037) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147476825) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)2032905037) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)2032905037) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147476825) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1118430789) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)2032905037) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)2032905037) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147476825) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)2032905037) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)2032905037) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147476825) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0)))) * (long)1118430789) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[16];
  __shared__ int placeholder_shared[1024];
  __shared__ int pad_data_shared[208];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 512) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + ((int)threadIdx.y)) < 74) {
        if ((((((int)blockIdx.y) * 32) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 4)) < 128) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((int)blockIdx.y) * 131072) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 57344)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 4) * 4096)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 15) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 63; ++ic_chunk_outer_outer) {
    __syncthreads();
    if (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) < 208) {
      if (((int)threadIdx.y) < 30) {
          ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 802816) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) / 52) * 200704)) + (ic_chunk_outer_outer * 3136)) + (((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) % 52) / 13) * 784)) + (((int)blockIdx.x) * 112)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) % 13) * 4)))))[0];
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 512) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + ((int)threadIdx.y)) < 74) {
          if ((((((int)blockIdx.y) * 32) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 4)) < 128) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 2048) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.y) * 131072) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 57344)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 4) * 4096)) + (ic_chunk_outer_outer * 64)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 15) * 4)) + 64))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 4; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 52) + (((int)threadIdx.x) * 8)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 64)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 52) + (((int)threadIdx.x) * 8)) + 208))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 64)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 52) + (((int)threadIdx.x) * 8)) + 416))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 64)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 52) + (((int)threadIdx.x) * 8)) + 624))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 2048) + (((int)threadIdx.y) * 64)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 12))]);
      }
    }
  }
  __syncthreads();
  if (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) < 208) {
    if (((int)threadIdx.y) < 30) {
        ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 28) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 802816) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) / 52) * 200704)) + (((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) % 52) / 13) * 784)) + (((int)blockIdx.x) * 112)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) % 13) * 4)) + 197568))))[0];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 4; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
      compute[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner1 * 52) + (((int)threadIdx.x) * 8)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 64) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute[(oc_block1)]);
      compute[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 52) + (((int)threadIdx.x) * 8)) + 208))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 64) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute[((oc_block1 + 4))]);
      compute[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 52) + (((int)threadIdx.x) * 8)) + 416))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 64) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute[((oc_block1 + 8))]);
      compute[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 52) + (((int)threadIdx.x) * 8)) + 624))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 64) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute[((oc_block1 + 12))]);
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[(((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741959) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741959) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2114189816) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741959) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741959) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2114189816) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 50176))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741959) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741959) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2114189816) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 75264))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741959) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1184596221) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741959) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2114189816) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__5_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[16];
  __shared__ int placeholder_shared[2048];
  __shared__ int pad_data_shared[896];
  #pragma unroll
  for (int n_init = 0; n_init < 2; ++n_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((n_init * 4) + oc_block_init))] = 0;
      compute[((((n_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 37) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((int)blockIdx.y) * 65536) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 4096)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 401408) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 200704)) + (ic_chunk_outer_outer * 12544)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 37) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.y) * 65536) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 4096)) + (ic_chunk_outer_outer * 256)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((n * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n * 1792) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((n * 4) + oc_block))]);
          compute[((((n * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n * 1792) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((n * 4) + oc_block) + 8))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 401408) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) >> 4) * 200704)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + 188160))))[0];
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int n1 = 0; n1 < 2; ++n1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((n1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 1792) + (ic_chunk_inner1 * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(((n1 * 4) + oc_block1))]);
        compute[((((n1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 1792) + (ic_chunk_inner1 * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((((n1 * 4) + oc_block1) + 8))]);
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 2; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1449996968) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1449996968) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147473765) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1449996968) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1449996968) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147473765) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1169894908) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1449996968) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1449996968) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147473765) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1449996968) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1449996968) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147473765) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0)))) * (long)1169894908) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_max_pool2d_1_kernel0(void* __restrict__ placeholder, void* __restrict__ tensor) {
  int tensor_local[1];
  tensor_local[(0)] = -2147483648;
  for (int rv = 0; rv < 3; ++rv) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor_local[(0)] = max((int)(tensor_local[(0)]), (int)((((1 <= ((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) % 3136) / 56) * 2) + rv)) && (1 <= (((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) % 56) * 2) + rv1))) ? (int)((int*)placeholder)[((((((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 56) * 896) + (rv * 448)) + ((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) % 56) * 8)) + (rv1 * 4)) + (((int)threadIdx.x) & 3)) - 452))] : (int)-2147483648)));
    }
  }
  ((int*)tensor)[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = tensor_local[(0)];
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_15119380522063600768__1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[16];
  __shared__ int pad_data_shared[1792];
  __shared__ int placeholder_shared[2048];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 50176)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)))))[0];
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 37) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 16384) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 3; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 3584) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) >> 4) * 50176)) + (ic_chunk_outer_outer * 12544)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + 12544))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 37) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 16384) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (ic_chunk_outer_outer * 256)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)) + 1792))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)) + 1792))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 12))]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
      compute[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 3584))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(oc_block1)]);
      compute[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 5376))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[((oc_block1 + 8))]);
      compute[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 3584))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((oc_block1 + 4))]);
      compute[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 5376))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((oc_block1 + 12))]);
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((int*)T_relu)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073767020) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073767020) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1367864638) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)1595052047) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073767020) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073767020) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1367864638) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))])) * (long)1595052047) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1073767020) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1073767020) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1367864638) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))])) * (long)1595052047) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1073767020) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1971949056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1073767020) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1367864638) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))])) * (long)1595052047) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ compute, void* __restrict__ placeholder2) {
  int compute1[28];
  __shared__ int pad_data_shared[440];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute1[(oc_block_init)] = 0;
    compute1[((oc_block_init + 4))] = 0;
    compute1[((oc_block_init + 8))] = 0;
    compute1[((oc_block_init + 12))] = 0;
    compute1[((oc_block_init + 16))] = 0;
    compute1[((oc_block_init + 20))] = 0;
    compute1[((oc_block_init + 24))] = 0;
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 8; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 4)) + ((int)threadIdx.x)) < 440) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + ((int)threadIdx.z)) < 110) {
            ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 802816) + (ic_chunk_outer * 100352)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 4)) + ((int)threadIdx.x)) / 55) * 12544)) + (((int)blockIdx.x) * 448)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 4)) + ((int)threadIdx.x)) % 55) * 4)))))[0];
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 32768) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 4096)) + ((((int)threadIdx.z) >> 3) * 1024)) + (ic_chunk_outer * 128)) + ((((int)threadIdx.z) & 7) * 16)) + (((int)threadIdx.x) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 8; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute1[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 220) + (((int)threadIdx.x) * 8)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[(oc_block)]);
        compute1[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 220) + (((int)threadIdx.x) * 8)) + 32))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 4))]);
        compute1[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 220) + (((int)threadIdx.x) * 8)) + 64))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 8))]);
        compute1[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 220) + (((int)threadIdx.x) * 8)) + 96))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 12))]);
        compute1[((oc_block + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 220) + (((int)threadIdx.x) * 8)) + 128))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 16))]);
        compute1[((oc_block + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 220) + (((int)threadIdx.x) * 8)) + 160))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 20))]);
        compute1[((oc_block + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 220) + (((int)threadIdx.x) * 8)) + 192))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 24))]);
      }
    }
  }
  #pragma unroll
  for (int i4 = 0; i4 < 4; ++i4) {
    ((int*)compute)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + i4))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[(i4)]) << ((long)18)) : ((long)compute1[(i4)])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[(i4)]) << ((long)18)) : ((long)compute1[(i4)])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1142567522) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + i4) + 16))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((i4 + 4))]) << ((long)18)) : ((long)compute1[((i4 + 4))])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((i4 + 4))]) << ((long)18)) : ((long)compute1[((i4 + 4))])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1142567522) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + i4) + 32))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((i4 + 8))]) << ((long)18)) : ((long)compute1[((i4 + 8))])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((i4 + 8))]) << ((long)18)) : ((long)compute1[((i4 + 8))])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1142567522) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + i4) + 48))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((i4 + 12))]) << ((long)18)) : ((long)compute1[((i4 + 12))])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((i4 + 12))]) << ((long)18)) : ((long)compute1[((i4 + 12))])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1142567522) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + i4) + 64))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((i4 + 16))]) << ((long)18)) : ((long)compute1[((i4 + 16))])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((i4 + 16))]) << ((long)18)) : ((long)compute1[((i4 + 16))])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1142567522) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + i4) + 80))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((i4 + 20))]) << ((long)18)) : ((long)compute1[((i4 + 20))])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((i4 + 20))]) << ((long)18)) : ((long)compute1[((i4 + 20))])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1142567522) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + i4) + 96))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((i4 + 24))]) << ((long)18)) : ((long)compute1[((i4 + 24))])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((i4 + 24))]) << ((long)18)) : ((long)compute1[((i4 + 24))])) * (long)1187934683) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + i4))]))) * (long)1142567522) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__1_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[16];
  __shared__ int pad_data_shared[864];
  __shared__ int placeholder_shared[1152];
  #pragma unroll
  for (int n_init = 0; n_init < 4; ++n_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((n_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 432) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 62) {
          ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448) + (((int)threadIdx.z) * 224)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) + ((int)blockIdx.x))) && ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) + ((int)blockIdx.x)) < 8)) && (1 <= (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 54) * 25088)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 54) / 27) * 196)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) * 28)) + (((int)blockIdx.x) * 28)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) * 4)) - 32))))[0] : (int)(int)0);
      }
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 576) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 83) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 11) {
            ((int*)((signed char*)placeholder_shared + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((int)threadIdx.z) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 72) * 18432)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 72) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((int)threadIdx.z) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 63; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 432) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 62) {
            ((int*)((signed char*)pad_data_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 1728) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.z) * 224)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) + ((int)blockIdx.x))) && ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) + ((int)blockIdx.x)) < 8)) && (1 <= (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? (int)((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 54) * 25088)) + (ic_chunk_outer_outer * 392)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 54) / 27) * 196)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) * 28)) + (((int)blockIdx.x) * 28)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) * 4)) + 360))))[0] : (int)(int)0);
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 576) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 83) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 11) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 2304) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 28) + (((int)threadIdx.z) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((((int)blockIdx.y) * 147456) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 72) * 18432)) + (ic_chunk_outer_outer * 288)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 72) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 28) + (((int)threadIdx.z) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)) + 288))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
      #pragma unroll
      for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
        #pragma unroll
        for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
          #pragma unroll
          for (int n = 0; n < 4; ++n) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[(((n * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 1728) + (((int)threadIdx.z) * 864)) + (n * 216)) + (ic_chunk_inner * 108)) + (kh_inner * 36)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 2304) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((n * 4) + oc_block))]);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
    #pragma unroll
    for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 2; ++ic_chunk_inner1) {
      #pragma unroll
      for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
        #pragma unroll
        for (int n1 = 0; n1 < 4; ++n1) {
          #pragma unroll
          for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
            compute[(((n1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 864) + (n1 * 216)) + (ic_chunk_inner1 * 108)) + (kh_inner1 * 36)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 1728))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 2304))))[0], compute[(((n1 * 4) + oc_block1))]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 4; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 100352)) + (ax0_inner_inner_inner_inner * 25088)) + (((int)blockIdx.y) * 1568)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1506044321) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1506044321) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1082269554) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_dense_add_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_add, void* __restrict__ placeholder2) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, float> T_dense_wmma_accumulator[1];
  __shared__ half compute_shared[512];
  __shared__ half compute_shared1[128];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half, nvcuda::wmma::col_major> compute_shared_wmma_matrix_b[1];
  (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000e+00f);
  for (int k_outer_outer = 0; k_outer_outer < 128; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 16; ++ax0_ax1_fused_outer_outer_outer_outer) {
      compute_shared[(((ax0_ax1_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = ((half)((float*)placeholder)[(((((ax0_ax1_fused_outer_outer_outer_outer * 4096) + ((((int)threadIdx.x) >> 4) * 2048)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)))]);
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 4; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      compute_shared1[(((ax0_ax1_fused_outer_outer_outer_outer1 * 32) + ((int)threadIdx.x)))] = ((half)((float*)placeholder1)[((((((((int)blockIdx.y) * 16384) + (ax0_ax1_fused_outer_outer_outer_outer1 * 4096)) + ((((int)threadIdx.x) >> 4) * 2048)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)))]);
    }
    __syncthreads();
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((half *)compute_shared + 0), 16);
    (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_b[0], ((half *)compute_shared1 + 0), 16);
    (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], compute_shared_wmma_matrix_a[0], compute_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
  }
  __syncthreads();
  (void)nvcuda::wmma::store_matrix_sync(((float *)compute_shared + 0), T_dense_wmma_accumulator[0], 8, nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int ax0_inner_ax1_inner_fused_outer_outer_outer_outer = 0; ax0_inner_ax1_inner_fused_outer_outer_outer_outer < 8; ++ax0_inner_ax1_inner_fused_outer_outer_outer_outer) {
    ((float*)T_add)[(((((ax0_inner_ax1_inner_fused_outer_outer_outer_outer * 4000) + ((((int)threadIdx.x) >> 3) * 1000)) + (((int)blockIdx.y) * 8)) + (((int)threadIdx.x) & 7)))] = (((float*)compute_shared)[(((ax0_inner_ax1_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] + ((float*)placeholder2)[(((((int)blockIdx.y) * 8) + (((int)threadIdx.x) & 7)))]);
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_3_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 1605632) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1595052047) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_10_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 49; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)2136391595) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_layout_transform_nn_batch_flatten_kernel0(void* __restrict__ tensor, void* __restrict__ placeholder) {
  ((float*)tensor)[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = ((float*)placeholder)[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))];
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_6_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 1605632) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1169394485) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__9_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[16];
  __shared__ int pad_data_shared[288];
  __shared__ int placeholder_shared[2304];
  #pragma unroll
  for (int oh_init = 0; oh_init < 2; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
      compute[((((oh_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 144) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 36) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 3) {
            ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6))) && ((((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6)) < 29)) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)))) && ((((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)) < 29)) ? (int)((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 72) * 100352)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 72) / 36) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6) * 4)) - 116))))[0] : (int)(int)0);
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 18) * 4608)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)))))[0];
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 144) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 36) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 3) {
              ((int*)((signed char*)pad_data_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 576) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6))) && ((((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6)) < 29)) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)))) && ((((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)) < 29)) ? (int)((int*)((signed char*)placeholder + ((((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 72) * 100352)) + (ic_chunk_outer_outer * 6272)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 72) / 36) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6) * 4)) + 6156))))[0] : (int)(int)0);
          }
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 4608) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 18) * 4608)) + (ic_chunk_outer_outer * 288)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)) + 288))))[0];
    }
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
        #pragma unroll
        for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
          #pragma unroll
          for (int oh = 0; oh < 2; ++oh) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 576) + (((int)threadIdx.z) * 288)) + (ic_chunk_inner * 144)) + (oh * 24)) + (kh_inner * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 4608) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
              compute[((((oh * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((ic_chunk_outer_outer & 1) * 576) + (((int)threadIdx.z) * 288)) + (ic_chunk_inner * 144)) + (oh * 24)) + (kh_inner * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 48))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 4608) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((((oh * 4) + oc_block) + 8))]);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 2; ++ic_chunk_inner1) {
      #pragma unroll
      for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
        #pragma unroll
        for (int oh1 = 0; oh1 < 2; ++oh1) {
          #pragma unroll
          for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
            compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 288) + (ic_chunk_inner1 * 144)) + (oh1 * 24)) + (kh_inner1 * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 576))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 4608))))[0], compute[(((oh1 * 4) + oc_block1))]);
            compute[((((oh1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 288) + (ic_chunk_inner1 * 144)) + (oh1 * 24)) + (kh_inner1 * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 624))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 4608))))[0], compute[((((oh1 * 4) + oc_block1) + 8))]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 100352)) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((17 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1366742087) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((17 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1366742087) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2126681667) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      ((signed char*)T_cast)[(((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 100352)) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((17 != 0) ? (((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))])) * (long)1366742087) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((17 != 0) ? (((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))])) * (long)1366742087) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2126681667) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_9_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 49; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1993989853) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_15119380522063600768__3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[16];
  __shared__ int pad_data_shared[1792];
  __shared__ int placeholder_shared[2048];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 50176)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)))))[0];
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 37) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 16384) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 3; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 3584) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) >> 4) * 50176)) + (ic_chunk_outer_outer * 12544)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + 12544))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 37) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 16384) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (ic_chunk_outer_outer * 256)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)) + 1792))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)) + 1792))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 12))]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
      compute[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 3584))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(oc_block1)]);
      compute[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 5376))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[((oc_block1 + 8))]);
      compute[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 3584))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((oc_block1 + 4))]);
      compute[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 5376))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((oc_block1 + 12))]);
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((int*)T_relu)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147430505) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147430505) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1786494896) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)2089225123) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147430505) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147430505) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1786494896) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))])) * (long)2089225123) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147430505) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147430505) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1786494896) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))])) * (long)2089225123) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147430505) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1098578370) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147430505) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1786494896) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))])) * (long)2089225123) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__13_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[16];
  __shared__ int pad_data_shared[512];
  __shared__ int placeholder_shared[1024];
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 4; ++ic_chunk_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 1605632) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 802816)) + (ic_chunk_outer * 200704)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)))))[0];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 2048) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (ic_chunk_outer * 256)) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
    }
    __syncthreads();
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)) + 1024))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)) + 1024))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 12))]);
      }
    }
  }
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))]))) * (long)1073763532) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.y) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))]))) * (long)1073763532) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.y) * 4) + ax4))])), (int)(0)))) * (long)1405843320) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + ax4) + 200704))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))]))) * (long)1073763532) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.y) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))]))) * (long)1073763532) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.y) * 4) + ax4))])), (int)(0)))) * (long)1405843320) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + ax4) + 100352))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))]))) * (long)1073763532) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)threadIdx.y) * 4) + ax4) + 32))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))]))) * (long)1073763532) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)threadIdx.y) * 4) + ax4) + 32))])), (int)(0)))) * (long)1405843320) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + ax4) + 301056))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))]))) * (long)1073763532) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)threadIdx.y) * 4) + ax4) + 32))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1538263139) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))]))) * (long)1073763532) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)threadIdx.y) * 4) + ax4) + 32))])), (int)(0)))) * (long)1405843320) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_7_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 1605632) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1837567025) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_15119380522063600768__10_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[16];
  __shared__ int pad_data_shared[128];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oc_chunk_init * 4) + oc_block_init))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ic_chunk_outer = 0; ic_chunk_outer < 2; ++ic_chunk_outer) {
    __syncthreads();
      ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ic_chunk_outer * 100352)) + ((((int)threadIdx.y) >> 1) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.y) & 1) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)))))[0];
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 8192) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 1024)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 2)) >> 3) * 256)) + (ic_chunk_outer * 128)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 2)) & 7) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 8; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((oc_chunk * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (oc_chunk * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((oc_chunk * 4) + oc_block))]);
          compute[((((oc_chunk * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)) + 32))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (oc_chunk * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((((oc_chunk * 4) + oc_block) + 8))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1481530374) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1481530374) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1073795082) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1481530374) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1481530374) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1073795082) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1134695180) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((1 != 0) ? (((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)1)) : ((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)1288667344) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
      ((int*)T_relu)[((((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1481530374) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1481530374) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1073795082) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1481530374) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1481530374) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1073795082) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1134695180) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((1 != 0) ? (((long)((int*)placeholder4)[((((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 224))]) << ((long)1)) : ((long)((int*)placeholder4)[((((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 224))])) * (long)1288667344) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_12402219635377536017__2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[32];
  __shared__ int pad_data_shared[3584];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oc_chunk_init * 4) + oc_block_init))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 16))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 8))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 24))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 100352) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 6272)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) >> 2) * 3136)) + (((int)blockIdx.x) * 448)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) & 3) * 112)) + ((((int)threadIdx.x) % 28) * 4)))))[0];
  }
  #pragma unroll
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 1; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)) + 7168))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 6272)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) >> 2) * 3136)) + (((int)blockIdx.x) * 448)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) & 3) * 112)) + ((((int)threadIdx.x) % 28) * 4)) + 50176))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 4) + ((int)threadIdx.y)) < 19) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 8192) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) >> 4) * 512)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((oc_chunk * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((oc_chunk * 4) + oc_block))]);
          compute[((((oc_chunk * 4) + oc_block) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((oc_chunk * 4) + oc_block) + 16))]);
          compute[((((oc_chunk * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)) + 224))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((((oc_chunk * 4) + oc_block) + 8))]);
          compute[((((oc_chunk * 4) + oc_block) + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)) + 224))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((oc_chunk * 4) + oc_block) + 24))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 4) + ((int)threadIdx.y)) < 19) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 8192) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) >> 4) * 512)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_chunk1 = 0; oc_chunk1 < 2; ++oc_chunk1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((oc_chunk1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7168))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oc_chunk1 * 4) + oc_block1))]);
        compute[((((oc_chunk1 * 4) + oc_block1) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7168))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute[((((oc_chunk1 * 4) + oc_block1) + 16))]);
        compute[((((oc_chunk1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7392))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((((oc_chunk1 * 4) + oc_block1) + 8))]);
        compute[((((oc_chunk1 * 4) + oc_block1) + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7392))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute[((((oc_chunk1 * 4) + oc_block1) + 24))]);
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1073758499) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1073758499) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2000488945) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4))])), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)1073758499) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)1073758499) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2000488945) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))])), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1073758499) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1073758499) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2000488945) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 224))])), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25312))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)1073758499) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1970177043) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)1073758499) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2000488945) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25312))])), (int)(0));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__7_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[28];
  __shared__ int pad_data_shared[256];
  __shared__ int placeholder_shared[576];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
    ((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= ((int)threadIdx.y)) && (((int)threadIdx.y) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) - 60))))[0] : (int)(int)0);
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 63; ++ic_chunk_outer_outer) {
    __syncthreads();
      ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= ((int)threadIdx.y)) && (((int)threadIdx.y) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (ic_chunk_outer_outer * 784)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) + 724))))[0] : (int)(int)0);
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 144) {
        if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 9) {
              ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 9) * 9216)) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 7; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 512) + (((int)threadIdx.z) * 256)) + ((((int)threadIdx.x) >> 1) * 112)) + (oh * 16)) + (kh_inner * 16)) + (kw_inner * 4)) + ((((int)threadIdx.x) & 1) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 144) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 9) {
            ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 9) * 9216)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)) + 9072))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 7; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 256) + ((((int)threadIdx.x) >> 1) * 112)) + (oh1 * 16)) + (kh_inner1 * 16)) + (kw_inner1 * 4)) + ((((int)threadIdx.x) & 1) * 4)) + 512))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + ((((int)threadIdx.x) >> 1) * 392)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1327850369) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1327850369) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1199626064) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ compute, void* __restrict__ placeholder2) {
  int compute1[28];
  __shared__ int pad_data_shared[1296];
  __shared__ int placeholder_shared[2048];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute1[(oc_block_init)] = 0;
    compute1[((oc_block_init + 4))] = 0;
    compute1[((oc_block_init + 8))] = 0;
    compute1[((oc_block_init + 12))] = 0;
    compute1[((oc_block_init + 16))] = 0;
    compute1[((oc_block_init + 20))] = 0;
    compute1[((oc_block_init + 24))] = 0;
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 8; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 11; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 1296) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + ((int)threadIdx.y)) < 324) {
            ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 401408) + (ic_chunk_outer * 50176)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 81) * 3136)) + (((int)blockIdx.x) * 448)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 81) / 27) * 112)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 27) * 4)))))[0];
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 16; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 65536) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 4096)) + ((((int)threadIdx.y) >> 4) * 2048)) + (ic_chunk_outer * 256)) + ((((int)threadIdx.y) & 15) * 16)) + (((int)threadIdx.x) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute1[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 324) + ((((int)threadIdx.x) >> 1) * 216)) + ((((int)threadIdx.x) & 1) * 8)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[(oc_block)]);
        compute1[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 324) + ((((int)threadIdx.x) >> 1) * 216)) + ((((int)threadIdx.x) & 1) * 8)) + 16))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 4))]);
        compute1[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 324) + ((((int)threadIdx.x) >> 1) * 216)) + ((((int)threadIdx.x) & 1) * 8)) + 32))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 8))]);
        compute1[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 324) + ((((int)threadIdx.x) >> 1) * 216)) + ((((int)threadIdx.x) & 1) * 8)) + 48))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 12))]);
        compute1[((oc_block + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 324) + ((((int)threadIdx.x) >> 1) * 216)) + ((((int)threadIdx.x) & 1) * 8)) + 64))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 16))]);
        compute1[((oc_block + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 324) + ((((int)threadIdx.x) >> 1) * 216)) + ((((int)threadIdx.x) & 1) * 8)) + 80))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 20))]);
        compute1[((oc_block + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 324) + ((((int)threadIdx.x) >> 1) * 216)) + ((((int)threadIdx.x) & 1) * 8)) + 96))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 24))]);
      }
    }
  }
  #pragma unroll
  for (int i4 = 0; i4 < 4; ++i4) {
    ((int*)compute)[((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + ((((int)threadIdx.x) >> 1) * 56)) + ((((int)threadIdx.x) & 1) * 4)) + i4))] = ((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute1[(i4)]) << ((long)17)) : ((long)compute1[(i4)])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute1[(i4)]) << ((long)17)) : ((long)compute1[(i4)])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))]))) * (long)1104238933) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + ((((int)threadIdx.x) >> 1) * 56)) + ((((int)threadIdx.x) & 1) * 4)) + i4) + 8))] = ((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute1[((i4 + 4))]) << ((long)17)) : ((long)compute1[((i4 + 4))])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute1[((i4 + 4))]) << ((long)17)) : ((long)compute1[((i4 + 4))])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))]))) * (long)1104238933) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + ((((int)threadIdx.x) >> 1) * 56)) + ((((int)threadIdx.x) & 1) * 4)) + i4) + 16))] = ((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute1[((i4 + 8))]) << ((long)17)) : ((long)compute1[((i4 + 8))])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute1[((i4 + 8))]) << ((long)17)) : ((long)compute1[((i4 + 8))])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))]))) * (long)1104238933) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + ((((int)threadIdx.x) >> 1) * 56)) + ((((int)threadIdx.x) & 1) * 4)) + i4) + 24))] = ((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute1[((i4 + 12))]) << ((long)17)) : ((long)compute1[((i4 + 12))])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute1[((i4 + 12))]) << ((long)17)) : ((long)compute1[((i4 + 12))])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))]))) * (long)1104238933) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + ((((int)threadIdx.x) >> 1) * 56)) + ((((int)threadIdx.x) & 1) * 4)) + i4) + 32))] = ((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute1[((i4 + 16))]) << ((long)17)) : ((long)compute1[((i4 + 16))])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute1[((i4 + 16))]) << ((long)17)) : ((long)compute1[((i4 + 16))])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))]))) * (long)1104238933) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + ((((int)threadIdx.x) >> 1) * 56)) + ((((int)threadIdx.x) & 1) * 4)) + i4) + 40))] = ((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute1[((i4 + 20))]) << ((long)17)) : ((long)compute1[((i4 + 20))])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute1[((i4 + 20))]) << ((long)17)) : ((long)compute1[((i4 + 20))])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))]))) * (long)1104238933) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + ((((int)threadIdx.x) >> 1) * 56)) + ((((int)threadIdx.x) & 1) * 4)) + i4) + 48))] = ((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute1[((i4 + 24))]) << ((long)17)) : ((long)compute1[((i4 + 24))])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute1[((i4 + 24))]) << ((long)17)) : ((long)compute1[((i4 + 24))])) * (long)1952966745) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + i4))]))) * (long)1104238933) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[16];
  __shared__ int pad_data_shared[864];
  __shared__ int placeholder_shared[1152];
  #pragma unroll
  for (int n_init = 0; n_init < 4; ++n_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((n_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 432) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 62) {
          ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448) + (((int)threadIdx.z) * 224)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) + ((int)blockIdx.x))) && ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) + ((int)blockIdx.x)) < 8)) && (1 <= (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 54) * 25088)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 54) / 27) * 196)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) * 28)) + (((int)blockIdx.x) * 28)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) * 4)) - 32))))[0] : (int)(int)0);
      }
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 576) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 83) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 11) {
            ((int*)((signed char*)placeholder_shared + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((int)threadIdx.z) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 72) * 18432)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 72) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((int)threadIdx.z) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 63; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 432) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 62) {
            ((int*)((signed char*)pad_data_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 1728) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.z) * 224)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) + ((int)blockIdx.x))) && ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) + ((int)blockIdx.x)) < 8)) && (1 <= (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? (int)((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 54) * 25088)) + (ic_chunk_outer_outer * 392)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 54) / 27) * 196)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) * 28)) + (((int)blockIdx.x) * 28)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) * 4)) + 360))))[0] : (int)(int)0);
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 576) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 83) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 11) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 2304) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 28) + (((int)threadIdx.z) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((((int)blockIdx.y) * 147456) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 72) * 18432)) + (ic_chunk_outer_outer * 288)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 72) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 28) + (((int)threadIdx.z) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)) + 288))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
      #pragma unroll
      for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
        #pragma unroll
        for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
          #pragma unroll
          for (int n = 0; n < 4; ++n) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[(((n * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 1728) + (((int)threadIdx.z) * 864)) + (n * 216)) + (ic_chunk_inner * 108)) + (kh_inner * 36)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 2304) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((n * 4) + oc_block))]);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
    #pragma unroll
    for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 2; ++ic_chunk_inner1) {
      #pragma unroll
      for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
        #pragma unroll
        for (int n1 = 0; n1 < 4; ++n1) {
          #pragma unroll
          for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
            compute[(((n1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 864) + (n1 * 216)) + (ic_chunk_inner1 * 108)) + (kh_inner1 * 36)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 1728))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 2304))))[0], compute[(((n1 * 4) + oc_block1))]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 4; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 100352)) + (ax0_inner_inner_inner_inner * 25088)) + (((int)blockIdx.y) * 1568)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1627237775) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1627237775) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1350577355) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 13; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 802816) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 3211264) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1576722265) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_15119380522063600768__8_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[32];
  __shared__ int pad_data_shared[3584];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oc_chunk_init * 4) + oc_block_init))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 16))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 8))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 24))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 100352) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 6272)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) >> 2) * 3136)) + (((int)blockIdx.x) * 448)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) & 3) * 112)) + ((((int)threadIdx.x) % 28) * 4)))))[0];
  }
  #pragma unroll
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 1; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)) + 7168))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 6272)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) >> 2) * 3136)) + (((int)blockIdx.x) * 448)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) & 3) * 112)) + ((((int)threadIdx.x) % 28) * 4)) + 50176))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 4) + ((int)threadIdx.y)) < 19) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 8192) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) >> 4) * 512)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((oc_chunk * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((oc_chunk * 4) + oc_block))]);
          compute[((((oc_chunk * 4) + oc_block) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((oc_chunk * 4) + oc_block) + 16))]);
          compute[((((oc_chunk * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)) + 224))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((((oc_chunk * 4) + oc_block) + 8))]);
          compute[((((oc_chunk * 4) + oc_block) + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)) + 224))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((oc_chunk * 4) + oc_block) + 24))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 4) + ((int)threadIdx.y)) < 19) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 8192) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) >> 4) * 512)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_chunk1 = 0; oc_chunk1 < 2; ++oc_chunk1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((oc_chunk1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7168))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oc_chunk1 * 4) + oc_block1))]);
        compute[((((oc_chunk1 * 4) + oc_block1) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7168))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute[((((oc_chunk1 * 4) + oc_block1) + 16))]);
        compute[((((oc_chunk1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7392))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((((oc_chunk1 * 4) + oc_block1) + 8))]);
        compute[((((oc_chunk1 * 4) + oc_block1) + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7392))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute[((((oc_chunk1 * 4) + oc_block1) + 24))]);
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147386978) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147386978) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1440775746) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)2059568007) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2147386978) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2147386978) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)1440775746) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))])) * (long)2059568007) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147386978) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147386978) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1440775746) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 224))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 224))])) * (long)2059568007) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25312))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2147386978) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)17)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1779395009) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2147386978) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)1440775746) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25312))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25312))])) * (long)2059568007) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__13_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[16];
  __shared__ int pad_data_shared[120];
  __shared__ int placeholder_shared[288];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 60) {
      ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 4) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 200704) + ((((int)blockIdx.x) / 7) * 896)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) - 228))))[0] : (int)(int)0);
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 60) {
        ((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer + 1) & 1) * 240) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 4) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ic_chunk_outer_outer * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) + 12316))))[0] : (int)(int)0);
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 72) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 64) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 288) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 36) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 256) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 18432) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304)) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 240) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
          compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 240) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 40))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 4))]);
          compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 240) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 80))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
          compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 240) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 120))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 12))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 72) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 64) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 288) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 36) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 256) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 18432) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2160))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_inner1 * 40) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 240))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(oc_block1)]);
        compute[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_inner1 * 40) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 280))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((oc_block1 + 4))]);
        compute[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_inner1 * 40) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 320))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((oc_block1 + 8))]);
        compute[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_inner1 * 40) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 360))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((oc_block1 + 12))]);
      }
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)2052399534) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)2052399534) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2044369894) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)2052399534) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)2052399534) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2044369894) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 448))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)2052399534) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)2052399534) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2044369894) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 672))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)2052399534) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)2052399534) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2044369894) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__10_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[16];
  __shared__ int pad_data_shared[288];
  __shared__ int placeholder_shared[2304];
  #pragma unroll
  for (int oh_init = 0; oh_init < 2; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
      compute[((((oh_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 144) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 36) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 3) {
            ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6))) && ((((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6)) < 29)) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)))) && ((((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)) < 29)) ? (int)((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 72) * 100352)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 72) / 36) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6) * 4)) - 116))))[0] : (int)(int)0);
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 18) * 4608)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)))))[0];
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 144) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 36) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 3) {
              ((int*)((signed char*)pad_data_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 576) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6))) && ((((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6)) < 29)) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)))) && ((((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)) < 29)) ? (int)((int*)((signed char*)placeholder + ((((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 72) * 100352)) + (ic_chunk_outer_outer * 6272)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 72) / 36) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6) * 4)) + 6156))))[0] : (int)(int)0);
          }
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 4608) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 18) * 4608)) + (ic_chunk_outer_outer * 288)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)) + 288))))[0];
    }
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
        #pragma unroll
        for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
          #pragma unroll
          for (int oh = 0; oh < 2; ++oh) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 576) + (((int)threadIdx.z) * 288)) + (ic_chunk_inner * 144)) + (oh * 24)) + (kh_inner * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 4608) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
              compute[((((oh * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((ic_chunk_outer_outer & 1) * 576) + (((int)threadIdx.z) * 288)) + (ic_chunk_inner * 144)) + (oh * 24)) + (kh_inner * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 48))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 4608) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((((oh * 4) + oc_block) + 8))]);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 2; ++ic_chunk_inner1) {
      #pragma unroll
      for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
        #pragma unroll
        for (int oh1 = 0; oh1 < 2; ++oh1) {
          #pragma unroll
          for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
            compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 288) + (ic_chunk_inner1 * 144)) + (oh1 * 24)) + (kh_inner1 * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 576))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 4608))))[0], compute[(((oh1 * 4) + oc_block1))]);
            compute[((((oh1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 288) + (ic_chunk_inner1 * 144)) + (oh1 * 24)) + (kh_inner1 * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 624))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 4608))))[0], compute[((((oh1 * 4) + oc_block1) + 8))]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 100352)) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((17 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1205664363) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((17 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1205664363) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2117040281) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      ((signed char*)T_cast)[(((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 100352)) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((17 != 0) ? (((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))])) * (long)1205664363) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((17 != 0) ? (((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))])) * (long)1205664363) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2117040281) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_15119380522063600768__5_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[16];
  __shared__ int pad_data_shared[1792];
  __shared__ int placeholder_shared[2048];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 50176)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)))))[0];
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 37) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 16384) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 3; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 3584) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) >> 4) * 50176)) + (ic_chunk_outer_outer * 12544)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + 12544))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 37) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 16384) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (ic_chunk_outer_outer * 256)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)) + 1792))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)) + 1792))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 12))]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
      compute[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 3584))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(oc_block1)]);
      compute[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 5376))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[((oc_block1 + 8))]);
      compute[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 3584))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((oc_block1 + 4))]);
      compute[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 5376))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((oc_block1 + 12))]);
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((int*)T_relu)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073788074) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073788074) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1113814462) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)1837567025) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073788074) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073788074) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1113814462) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))])) * (long)1837567025) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1073788074) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1073788074) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1113814462) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))])) * (long)1837567025) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1073788074) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1222145805) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1073788074) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1113814462) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))])) * (long)1837567025) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__8_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[28];
  __shared__ int pad_data_shared[896];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 12))] = 0;
    compute[((oc_block_init + 16))] = 0;
    compute[((oc_block_init + 20))] = 0;
    compute[((oc_block_init + 24))] = 0;
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 8; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 401408) + (ic_chunk_outer * 50176)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 56) * 3136)) + (((int)blockIdx.x) * 224)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 56) * 4)))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 32768) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 4096)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 4) * 2048)) + (ic_chunk_outer * 256)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 15) * 16)) + (((int)threadIdx.x) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 16))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 32))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 48))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 12))]);
        compute[((oc_block + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 64))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 16))]);
        compute[((oc_block + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 80))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 20))]);
        compute[((oc_block + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 96))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 24))]);
      }
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1150021518) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 16))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1150021518) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 32))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1150021518) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 48))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1150021518) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 64))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)16)) : ((long)compute[((ax4 + 16))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)16)) : ((long)compute[((ax4 + 16))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)16)) : ((long)compute[((ax4 + 16))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)16)) : ((long)compute[((ax4 + 16))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1150021518) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 80))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)16)) : ((long)compute[((ax4 + 20))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)16)) : ((long)compute[((ax4 + 20))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)16)) : ((long)compute[((ax4 + 20))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)16)) : ((long)compute[((ax4 + 20))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1150021518) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 96))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)16)) : ((long)compute[((ax4 + 24))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)16)) : ((long)compute[((ax4 + 24))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)16)) : ((long)compute[((ax4 + 24))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)16)) : ((long)compute[((ax4 + 24))])) * (long)1684117788) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147475627) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1150021518) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__10_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[28];
  __shared__ int pad_data_shared[896];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 12))] = 0;
    compute[((oc_block_init + 16))] = 0;
    compute[((oc_block_init + 20))] = 0;
    compute[((oc_block_init + 24))] = 0;
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 8; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 401408) + (ic_chunk_outer * 50176)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 56) * 3136)) + (((int)blockIdx.x) * 224)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 56) * 4)))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 32768) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 4096)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) >> 4) * 2048)) + (ic_chunk_outer * 256)) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.y)) & 15) * 16)) + (((int)threadIdx.x) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 16))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 32))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 48))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 12))]);
        compute[((oc_block + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 64))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 16))]);
        compute[((oc_block + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 80))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 20))]);
        compute[((oc_block + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner * 224) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + 96))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 24))]);
      }
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1532402649) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 16))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1532402649) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 32))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1532402649) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 48))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1532402649) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 64))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1532402649) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 80))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1532402649) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 96))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1522931676) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473483) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1532402649) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_15119380522063600768__4_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[16];
  __shared__ int pad_data_shared[1792];
  __shared__ int placeholder_shared[2048];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 50176)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)))))[0];
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 37) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 16384) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 3; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 3584) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) >> 4) * 50176)) + (ic_chunk_outer_outer * 12544)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + 12544))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 37) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 16384) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (ic_chunk_outer_outer * 256)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)) + 1792))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)) + 1792))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 12))]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
      compute[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 3584))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(oc_block1)]);
      compute[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 5376))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[((oc_block1 + 8))]);
      compute[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 3584))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((oc_block1 + 4))]);
      compute[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 5376))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((oc_block1 + 12))]);
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((int*)T_relu)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147411797) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147411797) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1787090516) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((1 != 0) ? (((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)1)) : ((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)1169394485) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147411797) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147411797) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1787090516) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((1 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))]) << ((long)1)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))])) * (long)1169394485) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147411797) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147411797) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1787090516) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((1 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))]) << ((long)1)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))])) * (long)1169394485) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147411797) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1376935399) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147411797) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1787090516) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((1 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))]) << ((long)1)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))])) * (long)1169394485) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_1_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 13; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 802816) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 3211264) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)2056266911) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__4_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[16];
  __shared__ int placeholder_shared[2048];
  __shared__ int pad_data_shared[896];
  #pragma unroll
  for (int n_init = 0; n_init < 2; ++n_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((n_init * 4) + oc_block_init))] = 0;
      compute[((((n_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 37) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((int)blockIdx.y) * 65536) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 4096)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 401408) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 200704)) + (ic_chunk_outer_outer * 12544)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 37) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.y) * 65536) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 4096)) + (ic_chunk_outer_outer * 256)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((n * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n * 1792) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((n * 4) + oc_block))]);
          compute[((((n * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n * 1792) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((n * 4) + oc_block) + 8))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 401408) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) >> 4) * 200704)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + 188160))))[0];
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int n1 = 0; n1 < 2; ++n1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((n1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 1792) + (ic_chunk_inner1 * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(((n1 * 4) + oc_block1))]);
        compute[((((n1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 1792) + (ic_chunk_inner1 * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((((n1 * 4) + oc_block1) + 8))]);
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 2; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1421641144) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1421641144) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147473910) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1421641144) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1421641144) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147473910) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1077860244) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1421641144) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1421641144) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147473910) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1421641144) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1421641144) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147473910) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0)))) * (long)1077860244) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_11_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 49; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)2059568007) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_15119380522063600768__2_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[16];
  __shared__ int pad_data_shared[1792];
  __shared__ int placeholder_shared[2048];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 50176)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)))))[0];
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 37) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 16384) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 3; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 3584) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) >> 4) * 50176)) + (ic_chunk_outer_outer * 12544)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + 12544))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 37) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 16384) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (ic_chunk_outer_outer * 256)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)) + 1792))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3584) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)) + 1792))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 12))]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
      compute[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 3584))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(oc_block1)]);
      compute[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 5376))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[((oc_block1 + 8))]);
      compute[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 3584))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((oc_block1 + 4))]);
      compute[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 112) + (((int)threadIdx.x) * 4)) + 5376))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((oc_block1 + 12))]);
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((int*)T_relu)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[(ax4)]) << ((long)19)) : ((long)compute[(ax4)])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[(ax4)]) << ((long)19)) : ((long)compute[(ax4)])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147438515) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[(ax4)]) << ((long)19)) : ((long)compute[(ax4)])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[(ax4)]) << ((long)19)) : ((long)compute[(ax4)])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147438515) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1640264770) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)1978965331) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)19)) : ((long)compute[((ax4 + 8))])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)19)) : ((long)compute[((ax4 + 8))])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147438515) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)19)) : ((long)compute[((ax4 + 8))])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)19)) : ((long)compute[((ax4 + 8))])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147438515) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1640264770) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 200704))])) * (long)1978965331) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)19)) : ((long)compute[((ax4 + 4))])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)19)) : ((long)compute[((ax4 + 4))])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147438515) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)19)) : ((long)compute[((ax4 + 4))])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)19)) : ((long)compute[((ax4 + 4))])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147438515) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1640264770) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))])) * (long)1978965331) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)19)) : ((long)compute[((ax4 + 12))])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)19)) : ((long)compute[((ax4 + 12))])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147438515) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)19)) : ((long)compute[((ax4 + 12))])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)19)) : ((long)compute[((ax4 + 12))])) * (long)1255636241) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147438515) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)1640264770) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 206976))])) * (long)1978965331) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__8_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[28];
  __shared__ int pad_data_shared[256];
  __shared__ int placeholder_shared[576];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
    ((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= ((int)threadIdx.y)) && (((int)threadIdx.y) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) - 60))))[0] : (int)(int)0);
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 63; ++ic_chunk_outer_outer) {
    __syncthreads();
      ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= ((int)threadIdx.y)) && (((int)threadIdx.y) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (ic_chunk_outer_outer * 784)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) + 724))))[0] : (int)(int)0);
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 144) {
        if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 9) {
              ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 9) * 9216)) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 7; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 512) + (((int)threadIdx.z) * 256)) + ((((int)threadIdx.x) >> 1) * 112)) + (oh * 16)) + (kh_inner * 16)) + (kw_inner * 4)) + ((((int)threadIdx.x) & 1) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 144) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 9) {
            ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 9) * 9216)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)) + 9072))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 7; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 256) + ((((int)threadIdx.x) >> 1) * 112)) + (oh1 * 16)) + (kh_inner1 * 16)) + (kw_inner1 * 4)) + ((((int)threadIdx.x) & 1) * 4)) + 512))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + ((((int)threadIdx.x) >> 1) * 392)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((15 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)15)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1733956305) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((15 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)15)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1733956305) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1659014097) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ compute, void* __restrict__ placeholder2) {
  int compute1[16];
  __shared__ int pad_data_shared[128];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute1[(((oc_chunk_init * 4) + oc_block_init))] = 0;
      compute1[((((oc_chunk_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ic_chunk_outer = 0; ic_chunk_outer < 2; ++ic_chunk_outer) {
    __syncthreads();
      ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ic_chunk_outer * 100352)) + ((((int)threadIdx.y) >> 1) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.y) & 1) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)))))[0];
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 8192) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 1024)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 2)) >> 3) * 256)) + (ic_chunk_outer * 128)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 2)) & 7) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 8; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute1[(((oc_chunk * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (oc_chunk * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[(((oc_chunk * 4) + oc_block))]);
          compute1[((((oc_chunk * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)) + 32))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (oc_chunk * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute1[((((oc_chunk * 4) + oc_block) + 8))]);
        }
      }
    }
  }
  #pragma unroll
  for (int i1_inner_inner_inner = 0; i1_inner_inner_inner < 2; ++i1_inner_inner_inner) {
    #pragma unroll
    for (int i4 = 0; i4 < 4; ++i4) {
      ((int*)compute)[(((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (i1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + i4))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[(((i1_inner_inner_inner * 4) + i4))]) << ((long)18)) : ((long)compute1[(((i1_inner_inner_inner * 4) + i4))])) * (long)1601687472) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (i1_inner_inner_inner * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[(((i1_inner_inner_inner * 4) + i4))]) << ((long)18)) : ((long)compute1[(((i1_inner_inner_inner * 4) + i4))])) * (long)1601687472) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (i1_inner_inner_inner * 4)) + i4))]))) * (long)1144303500) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
      ((int*)compute)[((((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (i1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + i4) + 224))] = ((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute1[((((i1_inner_inner_inner * 4) + i4) + 8))]) << ((long)18)) : ((long)compute1[((((i1_inner_inner_inner * 4) + i4) + 8))])) * (long)1601687472) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (i1_inner_inner_inner * 4)) + i4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute1[((((i1_inner_inner_inner * 4) + i4) + 8))]) << ((long)18)) : ((long)compute1[((((i1_inner_inner_inner * 4) + i4) + 8))])) * (long)1601687472) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (i1_inner_inner_inner * 4)) + i4))]))) * (long)1144303500) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_add_cast_nn_relu_cast_fixed_p_13766817817639592840__kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[32];
  __shared__ int pad_data_shared[224];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 16))] = 0;
    compute[((oc_block_init + 24))] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 12))] = 0;
    compute[((oc_block_init + 20))] = 0;
    compute[((oc_block_init + 28))] = 0;
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 64; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 401408) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 200704)) + ((((int)threadIdx.y) >> 3) * 100352)) + (ic_chunk_outer * 1568)) + ((((int)threadIdx.y) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 10; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.y)) < 147) {
            ((int*)((signed char*)placeholder_shared + ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 5) * 128) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 7) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 262144) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 5) * 8192)) + (ic_chunk_outer * 128)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 7) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 8; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)) + 224))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)) + 448))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 16))]);
        compute[((oc_block + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)) + 672))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 24))]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)) + 224))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 12))]);
        compute[((oc_block + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)) + 448))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 20))]);
        compute[((oc_block + 28))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)) + 672))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 28))]);
      }
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[(((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1145587898) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1145587898) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 50176))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1145587898) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 75264))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1145587898) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 3136))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0))) << ((long)0)) : ((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0)))) * (long)1145587898) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 28224))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0))) << ((long)0)) : ((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0)))) * (long)1145587898) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 53312))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0))) << ((long)0)) : ((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0)))) * (long)1145587898) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 78400))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 28))]) << ((long)17)) : ((long)compute[((ax4 + 28))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0))) << ((long)0)) : ((long)max((int)(((((int)(((((17 != 0) ? (((long)compute[((ax4 + 28))]) << ((long)17)) : ((long)compute[((ax4 + 28))])) * (long)1292030419) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0)))) * (long)1145587898) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__12_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[16];
  __shared__ int pad_data_shared[512];
  __shared__ int placeholder_shared[1024];
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 4; ++ic_chunk_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 1605632) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 802816)) + (ic_chunk_outer * 200704)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)))))[0];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 2048) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 2)) >> 4) * 1024)) + (ic_chunk_outer * 256)) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
    }
    __syncthreads();
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)) + 1024))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)) + 1024))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 12))]);
      }
    }
  }
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))]))) * (long)1073777777) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.y) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))]))) * (long)1073777777) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.y) * 4) + ax4))])), (int)(0)))) * (long)2032294944) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + ax4) + 200704))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))]))) * (long)1073777777) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.y) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.y) * 4) + ax4))]))) * (long)1073777777) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.y) * 4) + ax4))])), (int)(0)))) * (long)2032294944) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + ax4) + 100352))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))]))) * (long)1073777777) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)threadIdx.y) * 4) + ax4) + 32))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))]))) * (long)1073777777) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)threadIdx.y) * 4) + ax4) + 32))])), (int)(0)))) * (long)2032294944) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 401408) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.x) >> 3) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + ax4) + 301056))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))]))) * (long)1073777777) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)threadIdx.y) * 4) + ax4) + 32))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1523474988) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)threadIdx.y) * 4) + ax4) + 32))]))) * (long)1073777777) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)threadIdx.y) * 4) + ax4) + 32))])), (int)(0)))) * (long)2032294944) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_divide_add_round_cast_clip_cast_nn_pad_layout_transform_kernel0(void* __restrict__ T_layout_trans, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 1605632) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_layout_trans)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = (((((int)threadIdx.x) & 3) < 3) ? (signed char)((signed char)max((int)(min((int)(((int)roundf((((float*)placeholder)[((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) / 50176) * 150528) + ((((int)threadIdx.x) & 3) * 50176)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) % 50176)))] * 4.861612e+01f)))), (int)(127))), (int)(-128))) : (signed char)(signed char)0);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__14_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[28];
  __shared__ int pad_data_shared[224];
  __shared__ int placeholder_shared[256];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 12))] = 0;
    compute[((oc_block_init + 16))] = 0;
    compute[((oc_block_init + 20))] = 0;
    compute[((oc_block_init + 24))] = 0;
  }
  #pragma unroll
  for (int ic_chunk_outer = 0; ic_chunk_outer < 4; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.x)) < 224) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.z)) < 28) {
            ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 200704) + (ic_chunk_outer * 50176)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.x)) / 56) * 12544)) + (((int)blockIdx.x) * 224)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.x)) % 56) * 4)))))[0];
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 2048) + ((((((int)threadIdx.z) * 2) + (((int)threadIdx.x) >> 2)) >> 2) * 256)) + (ic_chunk_outer * 64)) + ((((((int)threadIdx.z) * 2) + (((int)threadIdx.x) >> 2)) & 3) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 4; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 224) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 64) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 224) + (((int)threadIdx.x) * 4)) + 32))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 64) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 224) + (((int)threadIdx.x) * 4)) + 64))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 64) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 224) + (((int)threadIdx.x) * 4)) + 96))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 64) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 12))]);
        compute[((oc_block + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 224) + (((int)threadIdx.x) * 4)) + 128))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 64) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 16))]);
        compute[((oc_block + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 224) + (((int)threadIdx.x) * 4)) + 160))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 64) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 20))]);
        compute[((oc_block + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 224) + (((int)threadIdx.x) * 4)) + 192))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 64) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 24))]);
      }
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(ax4)]) << ((long)17)) : ((long)compute[(ax4)])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0)))) * (long)1459213935) + ((long)1 << ((long)((22 + 31) - 1)))) >> ((long)(22 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + ax4) + 32))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)17)) : ((long)compute[((ax4 + 4))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0)))) * (long)1459213935) + ((long)1 << ((long)((22 + 31) - 1)))) >> ((long)(22 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + ax4) + 64))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)17)) : ((long)compute[((ax4 + 8))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0)))) * (long)1459213935) + ((long)1 << ((long)((22 + 31) - 1)))) >> ((long)(22 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + ax4) + 96))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)17)) : ((long)compute[((ax4 + 12))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0)))) * (long)1459213935) + ((long)1 << ((long)((22 + 31) - 1)))) >> ((long)(22 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + ax4) + 128))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)17)) : ((long)compute[((ax4 + 16))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0)))) * (long)1459213935) + ((long)1 << ((long)((22 + 31) - 1)))) >> ((long)(22 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + ax4) + 160))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)17)) : ((long)compute[((ax4 + 20))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0)))) * (long)1459213935) + ((long)1 << ((long)((22 + 31) - 1)))) >> ((long)(22 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + ax4) + 192))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))])) << ((long)1)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)17)) : ((long)compute[((ax4 + 24))])) * (long)1835168984) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + ax4))]))) * (long)1073745048) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((int)threadIdx.z) * 4) + ax4))])), (int)(0)))) * (long)1459213935) + ((long)1 << ((long)((22 + 31) - 1)))) >> ((long)(22 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_12402219635377536017__kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[28];
  __shared__ int pad_data_shared[1568];
  __shared__ int placeholder_shared[2048];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((int)blockIdx.z) * 25088) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0];
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 10; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 1024) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.y)) < 147) {
          ((int*)((signed char*)placeholder_shared + ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 6) * 256) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 15) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 32768) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 6) * 2048)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 15) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0];
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 7; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 3136) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 25088) + (ic_chunk_outer_outer * 3136)) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)) + 3136))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 10; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + ((int)threadIdx.y)) < 147) {
            ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 6) * 256)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 15) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 32768) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 6) * 2048)) + (ic_chunk_outer_outer * 256)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 15) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)) + 256))))[0];
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oh = 0; oh < 7; ++oh) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3136) + (ic_chunk_inner * 196)) + (oh * 28)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oh1 = 0; oh1 < 7; ++oh1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner1 * 196) + (oh1 * 28)) + (((int)threadIdx.x) * 4)) + 3136))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(((oh1 * 4) + oc_block1))]);
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[(((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (ax2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1173567685) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1173567685) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073742008) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1173567685) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1173567685) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073742008) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1592540765) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder4)[(((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (ax2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + ax4))])), (int)(0));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__6_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[28];
  __shared__ int pad_data_shared[256];
  __shared__ int placeholder_shared[576];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
    ((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= ((int)threadIdx.y)) && (((int)threadIdx.y) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) - 60))))[0] : (int)(int)0);
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 63; ++ic_chunk_outer_outer) {
    __syncthreads();
      ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= ((int)threadIdx.y)) && (((int)threadIdx.y) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (ic_chunk_outer_outer * 784)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) + 724))))[0] : (int)(int)0);
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 144) {
        if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 9) {
              ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 9) * 9216)) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 7; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 512) + (((int)threadIdx.z) * 256)) + ((((int)threadIdx.x) >> 1) * 112)) + (oh * 16)) + (kh_inner * 16)) + (kw_inner * 4)) + ((((int)threadIdx.x) & 1) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 144) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 9) {
            ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 9) * 9216)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)) + 9072))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 7; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 256) + ((((int)threadIdx.x) >> 1) * 112)) + (oh1 * 16)) + (kh_inner1 * 16)) + (kw_inner1 * 4)) + ((((int)threadIdx.x) & 1) * 4)) + 512))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + ((((int)threadIdx.x) >> 1) * 392)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1571730312) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1571730312) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2130801582) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__14_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[16];
  __shared__ int pad_data_shared[120];
  __shared__ int placeholder_shared[288];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 60) {
      ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 4) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 200704) + ((((int)blockIdx.x) / 7) * 896)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) - 228))))[0] : (int)(int)0);
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 60) {
        ((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer + 1) & 1) * 240) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 4) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ic_chunk_outer_outer * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) + 12316))))[0] : (int)(int)0);
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 72) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 64) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 288) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 36) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 256) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 18432) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304)) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 240) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
          compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 240) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 40))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 4))]);
          compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 240) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 80))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
          compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 240) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 120))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 12))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 72) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 64) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 288) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 36) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 256) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 18432) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2160))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_inner1 * 40) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 240))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(oc_block1)]);
        compute[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_inner1 * 40) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 280))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((oc_block1 + 4))]);
        compute[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_inner1 * 40) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 320))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((oc_block1 + 8))]);
        compute[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_inner1 * 40) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 360))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((oc_block1 + 12))]);
      }
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)1423657657) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)1423657657) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1105044667) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)1423657657) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)1423657657) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1105044667) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 448))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)1423657657) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)1423657657) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1105044667) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 672))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)1423657657) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)1423657657) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1105044667) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_13_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 98; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1117547056) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__7_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[28];
  __shared__ int pad_data_shared[1296];
  __shared__ int placeholder_shared[2048];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 12))] = 0;
    compute[((oc_block_init + 16))] = 0;
    compute[((oc_block_init + 20))] = 0;
    compute[((oc_block_init + 24))] = 0;
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) < 648) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 64) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.y)) < 324) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + ((int)threadIdx.z)) < 162) {
            ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((int)blockIdx.z) * 401408) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) / 81) * 3136)) + (((int)blockIdx.x) * 448)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) % 81) / 27) * 112)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) % 27) * 4)))))[0];
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)placeholder_shared + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) >> 5) * 128)) + ((((int)threadIdx.z) & 7) * 16)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 65536) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8192)) + (((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) >> 5) * 2048)) + ((((int)threadIdx.z) & 7) * 16)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)))))[0];
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) < 648) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 64) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.y)) < 324) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 32) + ((int)threadIdx.z)) < 162) {
              ((int*)((signed char*)pad_data_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 2592) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 16)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 401408) + (ic_chunk_outer_outer * 25088)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) / 81) * 3136)) + (((int)blockIdx.x) * 448)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) % 81) / 27) * 112)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) % 27) * 4)) + 25088))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)placeholder_shared + (((((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512)) + (((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) >> 5) * 128)) + ((((int)threadIdx.z) & 7) * 16)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((((int)blockIdx.y) * 65536) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8192)) + (((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) >> 5) * 2048)) + (ic_chunk_outer_outer * 128)) + ((((int)threadIdx.z) & 7) * 16)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 4)) + 128))))[0];
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 8; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 2592) + (ic_chunk_inner * 324)) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.z) * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 2592) + (ic_chunk_inner * 324)) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 16))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.z) * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 2592) + (ic_chunk_inner * 324)) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 32))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.z) * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 2592) + (ic_chunk_inner * 324)) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 48))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.z) * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 12))]);
        compute[((oc_block + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 2592) + (ic_chunk_inner * 324)) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 64))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.z) * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 16))]);
        compute[((oc_block + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 2592) + (ic_chunk_inner * 324)) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 80))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.z) * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 20))]);
        compute[((oc_block + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 2592) + (ic_chunk_inner * 324)) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 96))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.z) * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 24))]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 8; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
      compute[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner1 * 324) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 2592))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 128) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(oc_block1)]);
      compute[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner1 * 324) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 2608))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 128) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[((oc_block1 + 4))]);
      compute[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner1 * 324) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 2624))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 128) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[((oc_block1 + 8))]);
      compute[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner1 * 324) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 2640))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 128) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[((oc_block1 + 12))]);
      compute[((oc_block1 + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner1 * 324) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 2656))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 128) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[((oc_block1 + 16))]);
      compute[((oc_block1 + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner1 * 324) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 2672))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 128) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[((oc_block1 + 20))]);
      compute[((oc_block1 + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner1 * 324) + (((int)threadIdx.y) * 216)) + (((int)threadIdx.x) * 8)) + 2688))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.z) * 128) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[((oc_block1 + 24))]);
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 50176) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.y) * 56)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(ax4)]) << ((long)18)) : ((long)compute[(ax4)])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1121749979) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 50176) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.y) * 56)) + (((int)threadIdx.x) * 4)) + ax4) + 8))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)18)) : ((long)compute[((ax4 + 4))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1121749979) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 50176) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.y) * 56)) + (((int)threadIdx.x) * 4)) + ax4) + 16))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)18)) : ((long)compute[((ax4 + 8))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1121749979) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 50176) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.y) * 56)) + (((int)threadIdx.x) * 4)) + ax4) + 24))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)18)) : ((long)compute[((ax4 + 12))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1121749979) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 50176) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.y) * 56)) + (((int)threadIdx.x) * 4)) + ax4) + 32))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)18)) : ((long)compute[((ax4 + 16))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)18)) : ((long)compute[((ax4 + 16))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)18)) : ((long)compute[((ax4 + 16))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)18)) : ((long)compute[((ax4 + 16))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1121749979) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 50176) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.y) * 56)) + (((int)threadIdx.x) * 4)) + ax4) + 40))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)18)) : ((long)compute[((ax4 + 20))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)18)) : ((long)compute[((ax4 + 20))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)18)) : ((long)compute[((ax4 + 20))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)18)) : ((long)compute[((ax4 + 20))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1121749979) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 50176) + (((int)blockIdx.y) * 25088)) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.y) * 56)) + (((int)threadIdx.x) * 4)) + ax4) + 48))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)18)) : ((long)compute[((ax4 + 24))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)18)) : ((long)compute[((ax4 + 24))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)18)) : ((long)compute[((ax4 + 24))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)18)) : ((long)compute[((ax4 + 24))])) * (long)1321667431) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))]))) * (long)2147473582) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.z) * 4)) + ax4))])), (int)(0)))) * (long)1121749979) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__4_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[28];
  __shared__ int pad_data_shared[256];
  __shared__ int placeholder_shared[576];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
    ((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= ((int)threadIdx.y)) && (((int)threadIdx.y) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) - 60))))[0] : (int)(int)0);
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 63; ++ic_chunk_outer_outer) {
    __syncthreads();
      ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= ((int)threadIdx.y)) && (((int)threadIdx.y) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (ic_chunk_outer_outer * 784)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) + 724))))[0] : (int)(int)0);
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 144) {
        if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 9) {
              ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 9) * 9216)) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 7; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 512) + (((int)threadIdx.z) * 256)) + ((((int)threadIdx.x) >> 1) * 112)) + (oh * 16)) + (kh_inner * 16)) + (kw_inner * 4)) + ((((int)threadIdx.x) & 1) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 144) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 9) {
            ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 9) * 9216)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)) + 9072))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 7; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 256) + ((((int)threadIdx.x) >> 1) * 112)) + (oh1 * 16)) + (kh_inner1 * 16)) + (kw_inner1 * 4)) + ((((int)threadIdx.x) & 1) * 4)) + 512))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + ((((int)threadIdx.x) >> 1) * 392)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1972495247) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1972495247) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1152190013) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_9206029716410883813__kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_multiply, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[28];
  __shared__ int pad_data_shared[1568];
  __shared__ int placeholder_shared[2048];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((int)blockIdx.z) * 25088) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0];
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 10; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 1024) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.y)) < 147) {
          ((int*)((signed char*)placeholder_shared + ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 6) * 256) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 15) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 32768) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 6) * 2048)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 15) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0];
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 7; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 3136) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 25088) + (ic_chunk_outer_outer * 3136)) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)) + 3136))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 10; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + ((int)threadIdx.y)) < 147) {
            ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 6) * 256)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 15) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 32768) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 6) * 2048)) + (ic_chunk_outer_outer * 256)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 15) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)) + 256))))[0];
        }
      }
    }
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oh = 0; oh < 7; ++oh) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 3136) + (ic_chunk_inner * 196)) + (oh * 28)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oh1 = 0; oh1 < 7; ++oh1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((ic_chunk_inner1 * 196) + (oh1 * 28)) + (((int)threadIdx.x) * 4)) + 3136))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(((oh1 * 4) + oc_block1))]);
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((float*)T_multiply)[(((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (ax2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = (((float)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)19)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1156257710) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)19)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1156257710) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147483386) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((19 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)19)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1156257710) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((19 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)19)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1156257710) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147483386) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2136531316) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (ax2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 3136)) + (((int)threadIdx.y) * 196)) + (ax2_inner_inner_inner * 28)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)2140100076) + ((long)1 << ((long)((4 + 31) - 1)))) >> ((long)(4 + 31)))))), (int)(0))) * 1.461206e-08f);
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_5_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 25; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) < 1605632) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 6422528) {
        ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)2089225123) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_6343854372805914660__kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ compute, void* __restrict__ placeholder2) {
  int compute1[28];
  __shared__ int pad_data_shared[234];
  __shared__ int placeholder_shared[448];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute1[(oc_block_init)] = 0;
    compute1[((oc_block_init + 4))] = 0;
    compute1[((oc_block_init + 8))] = 0;
    compute1[((oc_block_init + 12))] = 0;
    compute1[((oc_block_init + 16))] = 0;
    compute1[((oc_block_init + 20))] = 0;
    compute1[((oc_block_init + 24))] = 0;
  }
  if (((((int)threadIdx.z) * 8) + ((int)threadIdx.x)) < 117) {
    if (((int)threadIdx.z) < 15) {
        ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 4)))))[0] = ((((4 <= ((int)blockIdx.x)) && (3 <= ((((((int)blockIdx.x) & 1) * 112) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.x)))) && (((((((int)blockIdx.x) & 1) * 112) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.x)) < 227)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 200704) + ((((int)blockIdx.x) >> 1) * 1792)) + ((((int)blockIdx.x) & 1) * 448)) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.x) * 4)) - 2700))))[0] : (int)(int)0);
    }
  }
  for (int kh_outer_outer = 0; kh_outer_outer < 6; ++kh_outer_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 8) + ((int)threadIdx.x)) < 117) {
      if (((int)threadIdx.z) < 15) {
          ((int*)((signed char*)pad_data_shared + ((((((kh_outer_outer + 1) & 1) * 468) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.x) * 4)))))[0] = (((((2 <= (((((int)blockIdx.x) >> 1) * 2) + kh_outer_outer)) && ((((((int)blockIdx.x) >> 1) * 2) + kh_outer_outer) < 226)) && (3 <= ((((((int)blockIdx.x) & 1) * 112) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.x)))) && (((((((int)blockIdx.x) & 1) * 112) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.x)) < 227)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + ((((int)blockIdx.x) >> 1) * 1792)) + (kh_outer_outer * 896)) + ((((int)blockIdx.x) & 1) * 448)) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.x) * 4)) - 1804))))[0] : (int)(int)0);
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 2)) + (((int)threadIdx.x) >> 2)) < 112) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.x)) < 448) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.z)) < 56) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 2)) + (((int)threadIdx.x) >> 2)) / 7) * 784) + (kh_outer_outer * 112)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 2)) + (((int)threadIdx.x) >> 2)) % 7) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 7; ++kw_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute1[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_outer_outer & 1) * 468) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute1[(oc_block)]);
        compute1[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((kh_outer_outer & 1) * 468) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)) + 64))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 4))]);
        compute1[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((kh_outer_outer & 1) * 468) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)) + 128))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 8))]);
        compute1[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((kh_outer_outer & 1) * 468) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)) + 192))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 12))]);
        compute1[((oc_block + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((kh_outer_outer & 1) * 468) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)) + 256))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 16))]);
        compute1[((oc_block + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((kh_outer_outer & 1) * 468) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)) + 320))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 20))]);
        compute1[((oc_block + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((kh_outer_outer & 1) * 468) + (((int)threadIdx.x) * 8)) + (kw_inner * 4)) + 384))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute1[((oc_block + 24))]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 2)) + (((int)threadIdx.x) >> 2)) < 112) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.x)) < 448) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + ((int)threadIdx.z)) < 56) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512) + (((int)threadIdx.z) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 2)) + (((int)threadIdx.x) >> 2)) / 7) * 784) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 2)) + (((int)threadIdx.x) >> 2)) % 7) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 672))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 7; ++kw_inner1) {
    #pragma unroll
    for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
      compute1[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + (((((int)threadIdx.x) * 8) + (kw_inner1 * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute1[(oc_block1)]);
      compute1[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.x) * 8) + (kw_inner1 * 4)) + 64))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute1[((oc_block1 + 4))]);
      compute1[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.x) * 8) + (kw_inner1 * 4)) + 128))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute1[((oc_block1 + 8))]);
      compute1[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.x) * 8) + (kw_inner1 * 4)) + 192))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute1[((oc_block1 + 12))]);
      compute1[((oc_block1 + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.x) * 8) + (kw_inner1 * 4)) + 256))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute1[((oc_block1 + 16))]);
      compute1[((oc_block1 + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.x) * 8) + (kw_inner1 * 4)) + 320))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute1[((oc_block1 + 20))]);
      compute1[((oc_block1 + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.x) * 8) + (kw_inner1 * 4)) + 384))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.z) * 112) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute1[((oc_block1 + 24))]);
    }
  }
  #pragma unroll
  for (int i4 = 0; i4 < 4; ++i4) {
    ((int*)compute)[((((((((int)blockIdx.z) * 802816) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + i4))] = ((int)(((((0 != 0) ? (((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[(i4)]) << ((long)15)) : ((long)compute1[(i4)])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[(i4)]) << ((long)15)) : ((long)compute1[(i4)])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0)))) * (long)2073482884) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[(((((((((int)blockIdx.z) * 802816) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + i4) + 32))] = ((int)(((((0 != 0) ? (((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((i4 + 4))]) << ((long)15)) : ((long)compute1[((i4 + 4))])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((i4 + 4))]) << ((long)15)) : ((long)compute1[((i4 + 4))])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0)))) * (long)2073482884) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[(((((((((int)blockIdx.z) * 802816) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + i4) + 64))] = ((int)(((((0 != 0) ? (((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((i4 + 8))]) << ((long)15)) : ((long)compute1[((i4 + 8))])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((i4 + 8))]) << ((long)15)) : ((long)compute1[((i4 + 8))])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0)))) * (long)2073482884) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[(((((((((int)blockIdx.z) * 802816) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + i4) + 96))] = ((int)(((((0 != 0) ? (((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((i4 + 12))]) << ((long)15)) : ((long)compute1[((i4 + 12))])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((i4 + 12))]) << ((long)15)) : ((long)compute1[((i4 + 12))])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0)))) * (long)2073482884) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[(((((((((int)blockIdx.z) * 802816) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + i4) + 128))] = ((int)(((((0 != 0) ? (((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((i4 + 16))]) << ((long)15)) : ((long)compute1[((i4 + 16))])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((i4 + 16))]) << ((long)15)) : ((long)compute1[((i4 + 16))])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0)))) * (long)2073482884) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[(((((((((int)blockIdx.z) * 802816) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + i4) + 160))] = ((int)(((((0 != 0) ? (((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((i4 + 20))]) << ((long)15)) : ((long)compute1[((i4 + 20))])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((i4 + 20))]) << ((long)15)) : ((long)compute1[((i4 + 20))])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0)))) * (long)2073482884) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
    ((int*)compute)[(((((((((int)blockIdx.z) * 802816) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.x) * 224)) + (((int)threadIdx.x) * 4)) + i4) + 192))] = ((int)(((((0 != 0) ? (((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((i4 + 24))]) << ((long)15)) : ((long)compute1[((i4 + 24))])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((15 != 0) ? (((long)compute1[((i4 + 24))]) << ((long)15)) : ((long)compute1[((i4 + 24))])) * (long)1398627452) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((int)threadIdx.z) * 4) + i4))])), (int)(0)))) * (long)2073482884) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31))));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[28];
  __shared__ int pad_data_shared[256];
  __shared__ int placeholder_shared[576];
  #pragma unroll
  for (int oh_init = 0; oh_init < 7; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
    }
  }
    ((int*)((signed char*)pad_data_shared + ((((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= ((int)threadIdx.y)) && (((int)threadIdx.y) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) - 60))))[0] : (int)(int)0);
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 63; ++ic_chunk_outer_outer) {
    __syncthreads();
      ((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= ((int)threadIdx.y)) && (((int)threadIdx.y) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x)))) && (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) < 15)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (ic_chunk_outer_outer * 784)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 4)) + 724))))[0] : (int)(int)0);
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 144) {
        if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 9) {
              ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 9) * 9216)) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oh = 0; oh < 7; ++oh) {
          #pragma unroll
          for (int oc_block = 0; oc_block < 4; ++oc_block) {
            compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 512) + (((int)threadIdx.z) * 256)) + ((((int)threadIdx.x) >> 1) * 112)) + (oh * 16)) + (kh_inner * 16)) + (kw_inner * 4)) + ((((int)threadIdx.x) & 1) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 144) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 576) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 9) {
            ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 9) * 9216)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 9) * 16)) + (((int)threadIdx.x) * 4)) + 9072))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oh1 = 0; oh1 < 7; ++oh1) {
        #pragma unroll
        for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
          compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 256) + ((((int)threadIdx.x) >> 1) * 112)) + (oh1 * 16)) + (kh_inner1 * 16)) + (kw_inner1 * 4)) + ((((int)threadIdx.x) & 1) * 4)) + 512))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oh1 * 4) + oc_block1))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 7; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + ((((int)threadIdx.x) >> 1) * 392)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((17 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1310439947) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((17 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1310439947) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2139368712) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__15_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[16];
  __shared__ int pad_data_shared[120];
  __shared__ int placeholder_shared[288];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 12))] = 0;
  }
  if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 60) {
      ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 4) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 200704) + ((((int)blockIdx.x) / 7) * 896)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) - 228))))[0] : (int)(int)0);
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 60) {
        ((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer + 1) & 1) * 240) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10))) && ((((((int)blockIdx.x) / 7) * 4) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 57)) && (1 <= (((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && ((((((int)blockIdx.x) % 7) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ic_chunk_outer_outer * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10) * 4)) + 12316))))[0] : (int)(int)0);
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 72) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 64) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 288) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 36) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 256) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 18432) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304)) + (ic_chunk_outer_outer * 144)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((ic_chunk_outer_outer & 1) * 240) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
          compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 240) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 40))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 4))]);
          compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 240) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 80))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
          compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((ic_chunk_outer_outer & 1) * 240) + (kh_inner * 40)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 120))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 12))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 72) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 64) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 288) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 36) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 256) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 18432) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 9) * 2304)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 9) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2160))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(oc_block1)] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_inner1 * 40) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 240))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(oc_block1)]);
        compute[((oc_block1 + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_inner1 * 40) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 280))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((oc_block1 + 4))]);
        compute[((oc_block1 + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_inner1 * 40) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 320))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((oc_block1 + 8))]);
        compute[((oc_block1 + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((kh_inner1 * 40) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 360))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 144) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((oc_block1 + 12))]);
      }
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)1347813883) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)1347813883) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1572057641) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)1347813883) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)1347813883) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1572057641) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 448))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)1347813883) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)1347813883) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1572057641) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (((int)blockIdx.y) * 100352)) + (((int)threadIdx.y) * 12544)) + ((((int)blockIdx.x) / 7) * 896)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 672))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)1347813883) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)1347813883) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1572057641) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[16];
  __shared__ int pad_data_shared[864];
  __shared__ int placeholder_shared[1152];
  #pragma unroll
  for (int n_init = 0; n_init < 4; ++n_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((n_init * 4) + oc_block_init))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 432) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 62) {
          ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448) + (((int)threadIdx.z) * 224)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) + ((int)blockIdx.x))) && ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) + ((int)blockIdx.x)) < 8)) && (1 <= (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? (int)((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 54) * 25088)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 54) / 27) * 196)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) * 28)) + (((int)blockIdx.x) * 28)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) * 4)) - 32))))[0] : (int)(int)0);
      }
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 576) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 83) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 11) {
            ((int*)((signed char*)placeholder_shared + (((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((int)threadIdx.z) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 147456) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 72) * 18432)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 72) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((int)threadIdx.z) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 63; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 432) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 62) {
            ((int*)((signed char*)pad_data_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 1728) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 448)) + (((int)threadIdx.z) * 224)) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) + ((int)blockIdx.x))) && ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) + ((int)blockIdx.x)) < 8)) && (1 <= (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? (int)((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 54) * 25088)) + (ic_chunk_outer_outer * 392)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 54) / 27) * 196)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 27) / 9) * 28)) + (((int)blockIdx.x) * 28)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) * 4)) + 360))))[0] : (int)(int)0);
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 576) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 16) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.y)) < 83) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 11) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 2304) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 28) + (((int)threadIdx.z) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((((int)blockIdx.y) * 147456) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 72) * 18432)) + (ic_chunk_outer_outer * 288)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 112) + (((int)threadIdx.z) * 56)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 72) / 12) * 48)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 28) + (((int)threadIdx.z) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) % 3) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)) + 288))))[0];
          }
        }
      }
    }
    #pragma unroll
    for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
      #pragma unroll
      for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
        #pragma unroll
        for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
          #pragma unroll
          for (int n = 0; n < 4; ++n) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[(((n * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 1728) + (((int)threadIdx.z) * 864)) + (n * 216)) + (ic_chunk_inner * 108)) + (kh_inner * 36)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 2304) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((n * 4) + oc_block))]);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
    #pragma unroll
    for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 2; ++ic_chunk_inner1) {
      #pragma unroll
      for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
        #pragma unroll
        for (int n1 = 0; n1 < 4; ++n1) {
          #pragma unroll
          for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
            compute[(((n1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 864) + (n1 * 216)) + (ic_chunk_inner1 * 108)) + (kh_inner1 * 36)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 1728))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 2304))))[0], compute[(((n1 * 4) + oc_block1))]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 4; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 100352)) + (ax0_inner_inner_inner_inner * 25088)) + (((int)blockIdx.y) * 1568)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1529959532) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1529959532) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 32) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2094848558) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_global_avg_pool2d_1_kernel0(void* __restrict__ placeholder, void* __restrict__ tensor) {
  float tensor1[4];
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    tensor1[(ax4)] = 0.000000e+00f;
    for (int rv0 = 0; rv0 < 7; ++rv0) {
      for (int rv1 = 0; rv1 < 7; ++rv1) {
        tensor1[(ax4)] = (tensor1[(ax4)] + ((float*)placeholder)[((((((((((int)blockIdx.y) * 802816) + (((int)threadIdx.y) * 100352)) + (((int)blockIdx.x) * 1568)) + (((int)threadIdx.x) * 196)) + (rv0 * 28)) + (rv1 * 4)) + ax4))]);
      }
    }
  }
  for (int ax41 = 0; ax41 < 4; ++ax41) {
    ((float*)tensor)[((((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 2048)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) * 4)) + ax41))] = (tensor1[(ax41)] * 2.040816e-02f);
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__3_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[16];
  __shared__ int placeholder_shared[2048];
  __shared__ int pad_data_shared[896];
  #pragma unroll
  for (int n_init = 0; n_init < 2; ++n_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((n_init * 4) + oc_block_init))] = 0;
      compute[((((n_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) < 37) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((int)blockIdx.y) * 65536) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 4096)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
        }
      }
    }
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 401408) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) >> 4) * 200704)) + (ic_chunk_outer_outer * 12544)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) < 37) {
              ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer + 1) & 1) * 4096) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.y) * 65536) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) >> 4) * 4096)) + (ic_chunk_outer_outer * 256)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 7)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((n * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n * 1792) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((n * 4) + oc_block))]);
          compute[((((n * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n * 1792) + (ic_chunk_inner * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((ic_chunk_outer_outer & 1) * 4096) + (((int)threadIdx.y) * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((n * 4) + oc_block) + 8))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.z) * 401408) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) >> 4) * 200704)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 8) + ((int)threadIdx.y)) & 15) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + 188160))))[0];
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int n1 = 0; n1 < 2; ++n1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((n1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 1792) + (ic_chunk_inner1 * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 4096))))[0], compute[(((n1 * 4) + oc_block1))]);
        compute[((((n1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((n1 * 1792) + (ic_chunk_inner1 * 112)) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 6144))))[0], compute[((((n1 * 4) + oc_block1) + 8))]);
      }
    }
  }
  #pragma unroll
  for (int ax0_inner_inner_inner_inner = 0; ax0_inner_inner_inner_inner < 2; ++ax0_inner_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1488572971) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1488572971) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147475317) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1488572971) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))]) << ((long)17)) : ((long)compute[(((ax0_inner_inner_inner_inner * 4) + ax4))])) * (long)1488572971) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)2147475317) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1085271558) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      ((signed char*)T_cast)[(((((((((((int)blockIdx.z) * 100352) + (ax0_inner_inner_inner_inner * 50176)) + (((int)blockIdx.y) * 12544)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.x) * 112)) + (((int)threadIdx.x) * 4)) + ax4) + 6272))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1488572971) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1488572971) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147475317) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1488572971) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((17 != 0) ? (((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))]) << ((long)17)) : ((long)compute[((((ax0_inner_inner_inner_inner * 4) + ax4) + 8))])) * (long)1488572971) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))]))) * (long)2147475317) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4) + 32))])), (int)(0)))) * (long)1085271558) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_16086763325481941859__kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2, void* __restrict__ placeholder3) {
  int compute[32];
  __shared__ int pad_data_shared[224];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
    compute[(oc_block_init)] = 0;
    compute[((oc_block_init + 8))] = 0;
    compute[((oc_block_init + 16))] = 0;
    compute[((oc_block_init + 24))] = 0;
    compute[((oc_block_init + 4))] = 0;
    compute[((oc_block_init + 12))] = 0;
    compute[((oc_block_init + 20))] = 0;
    compute[((oc_block_init + 28))] = 0;
  }
  for (int ic_chunk_outer = 0; ic_chunk_outer < 64; ++ic_chunk_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 448) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 401408) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 200704)) + ((((int)threadIdx.y) >> 3) * 100352)) + (ic_chunk_outer * 1568)) + ((((int)threadIdx.y) & 7) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 10; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 16) + ((int)threadIdx.y)) < 147) {
            ((int*)((signed char*)placeholder_shared + ((((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 5) * 128) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 7) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 262144) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) >> 5) * 8192)) + (ic_chunk_outer * 128)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 28) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) >> 2)) & 7) * 16)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) & 3) * 4)))))[0];
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 8; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_block = 0; oc_block < 4; ++oc_block) {
        compute[(oc_block)] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(oc_block)]);
        compute[((oc_block + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)) + 224))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 8))]);
        compute[((oc_block + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)) + 448))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 16))]);
        compute[((oc_block + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)) + 672))))[0], ((int*)((signed char*)placeholder_shared + ((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((oc_block + 24))]);
        compute[((oc_block + 4))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 4))]);
        compute[((oc_block + 12))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)) + 224))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 12))]);
        compute[((oc_block + 20))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)) + 448))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 20))]);
        compute[((oc_block + 28))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 28) + (((int)threadIdx.x) * 4)) + 672))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 128) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((oc_block + 28))]);
      }
    }
  }
  #pragma unroll
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    ((signed char*)T_cast)[(((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[(ax4)]) << ((long)16)) : ((long)compute[(ax4)])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2131207242) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 8))]) << ((long)16)) : ((long)compute[((ax4 + 8))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2131207242) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 50176))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)16)) : ((long)compute[((ax4 + 16))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)16)) : ((long)compute[((ax4 + 16))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)16)) : ((long)compute[((ax4 + 16))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 16))]) << ((long)16)) : ((long)compute[((ax4 + 16))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2131207242) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 75264))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)16)) : ((long)compute[((ax4 + 24))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)16)) : ((long)compute[((ax4 + 24))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)16)) : ((long)compute[((ax4 + 24))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 24))]) << ((long)16)) : ((long)compute[((ax4 + 24))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)2131207242) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 3136))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 4))]) << ((long)16)) : ((long)compute[((ax4 + 4))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0)))) * (long)2131207242) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 28224))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 12))]) << ((long)16)) : ((long)compute[((ax4 + 12))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0)))) * (long)2131207242) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 53312))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)16)) : ((long)compute[((ax4 + 20))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)16)) : ((long)compute[((ax4 + 20))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)16)) : ((long)compute[((ax4 + 20))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 20))]) << ((long)16)) : ((long)compute[((ax4 + 20))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0)))) * (long)2131207242) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
    ((signed char*)T_cast)[((((((((((int)blockIdx.z) * 100352) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 196)) + (((int)blockIdx.x) * 28)) + (((int)threadIdx.x) * 4)) + ax4) + 78400))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 28))]) << ((long)16)) : ((long)compute[((ax4 + 28))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 28))]) << ((long)16)) : ((long)compute[((ax4 + 28))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((1 != 0) ? (((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 28))]) << ((long)16)) : ((long)compute[((ax4 + 28))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])) << ((long)1)) : ((long)(((int)(((((16 != 0) ? (((long)compute[((ax4 + 28))]) << ((long)16)) : ((long)compute[((ax4 + 28))])) * (long)2054051438) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))]))) * (long)1073741938) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 4)) + ax4) + 64))])), (int)(0)))) * (long)2131207242) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_12_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 98; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)2127229536) + ((long)1 << ((long)((24 + 31) - 1)))) >> ((long)(24 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_15119380522063600768__9_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[16];
  __shared__ int pad_data_shared[128];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oc_chunk_init * 4) + oc_block_init))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ic_chunk_outer = 0; ic_chunk_outer < 2; ++ic_chunk_outer) {
    __syncthreads();
      ((int*)((signed char*)pad_data_shared + (((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 200704) + (ic_chunk_outer * 100352)) + ((((int)threadIdx.y) >> 1) * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)threadIdx.y) & 1) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)))))[0];
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
        ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 8192) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 1024)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 2)) >> 3) * 256)) + (ic_chunk_outer * 128)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 2)) & 7) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 8; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((oc_chunk * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (oc_chunk * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((oc_chunk * 4) + oc_block))]);
          compute[((((oc_chunk * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 64) + (((int)threadIdx.x) * 4)) + 32))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 256) + (oc_chunk * 128)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((((oc_chunk * 4) + oc_block) + 8))]);
        }
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1237566260) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1237566260) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1073846162) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1237566260) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1237566260) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1073846162) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2009407807) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((1 != 0) ? (((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)1)) : ((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)1117547056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
      ((int*)T_relu)[((((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1237566260) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1237566260) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1073846162) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((1 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1237566260) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)1)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1237566260) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1073846162) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2009407807) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((1 != 0) ? (((long)((int*)placeholder4)[((((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 224))]) << ((long)1)) : ((long)((int*)placeholder4)[((((((((((((int)blockIdx.z) * 802816) + (((int)blockIdx.y) * 401408)) + (((int)threadIdx.y) * 25088)) + (ax1_inner_inner_inner * 12544)) + ((((int)blockIdx.x) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) * 4)) + ax4) + 224))])) * (long)1117547056) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_nn_relu_cast_fixed_point_mult_18399029763786111876__12_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_cast, void* __restrict__ placeholder2) {
  int compute[16];
  __shared__ int pad_data_shared[288];
  __shared__ int placeholder_shared[2304];
  #pragma unroll
  for (int oh_init = 0; oh_init < 2; ++oh_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oh_init * 4) + oc_block_init))] = 0;
      compute[((((oh_init * 4) + oc_block_init) + 8))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
    if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 144) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 36) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 2) + ((int)threadIdx.z)) < 3) {
            ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6))) && ((((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6)) < 29)) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)))) && ((((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)) < 29)) ? (int)((int*)((signed char*)placeholder + (((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 72) * 100352)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 72) / 36) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6) * 4)) - 116))))[0] : (int)(int)0);
        }
      }
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)placeholder_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 18) * 4608)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)))))[0];
  }
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 15; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) < 144) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) < 36) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 2) + ((int)threadIdx.z)) < 3) {
              ((int*)((signed char*)pad_data_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 576) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = (((((1 <= (((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6))) && ((((((int)blockIdx.x) / 7) * 4) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6)) < 29)) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)))) && ((((((int)blockIdx.x) % 7) * 4) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6)) < 29)) ? (int)((int*)((signed char*)placeholder + ((((((((((((int)blockIdx.z) * 200704) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) / 72) * 100352)) + (ic_chunk_outer_outer * 6272)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 72) / 36) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 36) / 6) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 128) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) % 6) * 4)) + 6156))))[0] : (int)(int)0);
          }
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 9; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer + 1) & 1) * 4608) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((((int)blockIdx.y) * 73728) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) / 18) * 4608)) + (ic_chunk_outer_outer * 288)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 32) + (((int)threadIdx.z) * 16)) + ((int)threadIdx.y)) % 18) * 16)) + (((int)threadIdx.x) * 4)) + 288))))[0];
    }
    #pragma unroll
    for (int kw_inner = 0; kw_inner < 3; ++kw_inner) {
      #pragma unroll
      for (int ic_chunk_inner = 0; ic_chunk_inner < 2; ++ic_chunk_inner) {
        #pragma unroll
        for (int kh_inner = 0; kh_inner < 3; ++kh_inner) {
          #pragma unroll
          for (int oh = 0; oh < 2; ++oh) {
            #pragma unroll
            for (int oc_block = 0; oc_block < 4; ++oc_block) {
              compute[(((oh * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((((((((ic_chunk_outer_outer & 1) * 576) + (((int)threadIdx.z) * 288)) + (ic_chunk_inner * 144)) + (oh * 24)) + (kh_inner * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 4608) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[(((oh * 4) + oc_block))]);
              compute[((((oh * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((ic_chunk_outer_outer & 1) * 576) + (((int)threadIdx.z) * 288)) + (ic_chunk_inner * 144)) + (oh * 24)) + (kh_inner * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner * 4)) + 48))))[0], ((int*)((signed char*)placeholder_shared + ((((((((ic_chunk_outer_outer & 1) * 4608) + (((int)threadIdx.y) * 288)) + (ic_chunk_inner * 144)) + (kh_inner * 48)) + (kw_inner * 16)) + (oc_block * 4)))))[0], compute[((((oh * 4) + oc_block) + 8))]);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int kw_inner1 = 0; kw_inner1 < 3; ++kw_inner1) {
    #pragma unroll
    for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 2; ++ic_chunk_inner1) {
      #pragma unroll
      for (int kh_inner1 = 0; kh_inner1 < 3; ++kh_inner1) {
        #pragma unroll
        for (int oh1 = 0; oh1 < 2; ++oh1) {
          #pragma unroll
          for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
            compute[(((oh1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 288) + (ic_chunk_inner1 * 144)) + (oh1 * 24)) + (kh_inner1 * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 576))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 4608))))[0], compute[(((oh1 * 4) + oc_block1))]);
            compute[((((oh1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((((((((int)threadIdx.z) * 288) + (ic_chunk_inner1 * 144)) + (oh1 * 24)) + (kh_inner1 * 24)) + (((int)threadIdx.x) * 4)) + (kw_inner1 * 4)) + 624))))[0], ((int*)((signed char*)placeholder_shared + (((((((((int)threadIdx.y) * 288) + (ic_chunk_inner1 * 144)) + (kh_inner1 * 48)) + (kw_inner1 * 16)) + (oc_block1 * 4)) + 4608))))[0], compute[((((oh1 * 4) + oc_block1) + 8))]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((signed char*)T_cast)[((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 100352)) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + ax4))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1565194612) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[(((ax2_inner_inner_inner * 4) + ax4))]) << ((long)16)) : ((long)compute[(((ax2_inner_inner_inner * 4) + ax4))])) * (long)1565194612) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1208910866) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
      ((signed char*)T_cast)[(((((((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 100352)) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 3136)) + ((((int)blockIdx.x) / 7) * 448)) + (ax2_inner_inner_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))]) << ((long)16)) : ((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))])) * (long)1565194612) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0))) << ((long)0)) : ((long)max((int)((((int)(((((16 != 0) ? (((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))]) << ((long)16)) : ((long)compute[((((ax2_inner_inner_inner * 4) + ax4) + 8))])) * (long)1565194612) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ax4))])), (int)(0)))) * (long)1208910866) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
    }
  }
}

extern "C" __global__ void fused_cast_fixed_point_multiply_clip_cast_14_kernel0(void* __restrict__ T_cast, void* __restrict__ placeholder) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer < 98; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer) {
    ((signed char*)T_cast)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((signed char)max((int)(min((int)(((int)(((((0 != 0) ? (((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))]) << ((long)0)) : ((long)((int*)placeholder)[((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))])) * (long)1460722492) + ((long)1 << ((long)((23 + 31) - 1)))) >> ((long)(23 + 31))))), (int)(127))), (int)(-128)));
  }
}

extern "C" __global__ void fused_nn_conv2d_cast_fixed_point_multiply_add_cast_fixed_point_multiply_add_cast_15119380522063600768__6_kernel0(void* __restrict__ placeholder, void* __restrict__ placeholder1, void* __restrict__ T_relu, void* __restrict__ placeholder2, void* __restrict__ placeholder3, void* __restrict__ placeholder4) {
  int compute[32];
  __shared__ int pad_data_shared[3584];
  __shared__ int placeholder_shared[1024];
  #pragma unroll
  for (int oc_chunk_init = 0; oc_chunk_init < 2; ++oc_chunk_init) {
    #pragma unroll
    for (int oc_block_init = 0; oc_block_init < 4; ++oc_block_init) {
      compute[(((oc_chunk_init * 4) + oc_block_init))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 16))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 8))] = 0;
      compute[((((oc_chunk_init * 4) + oc_block_init) + 24))] = 0;
    }
  }
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer) {
      ((int*)((signed char*)pad_data_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder + (((((((((int)blockIdx.z) * 100352) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer * 6272)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) >> 2) * 3136)) + (((int)blockIdx.x) * 448)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) & 3) * 112)) + ((((int)threadIdx.x) % 28) * 4)))))[0];
  }
  #pragma unroll
  for (int ic_chunk_outer_outer = 0; ic_chunk_outer_outer < 1; ++ic_chunk_outer_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1) {
        ((int*)((signed char*)pad_data_shared + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)) + 7168))))[0] = ((int*)((signed char*)placeholder + ((((((((((int)blockIdx.z) * 100352) + (ax0_ax1_fused_ax2_fused_ax3_fused_ax4_outer_fused_outer_outer_outer1 * 6272)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) >> 2) * 3136)) + (((int)blockIdx.x) * 448)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 28)) & 3) * 112)) + ((((int)threadIdx.x) % 28) * 4)) + 50176))))[0];
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) < 256) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 224) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) < 1024) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 4) + ((int)threadIdx.y)) < 19) {
              ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + (((((((int)blockIdx.y) * 8192) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) >> 4) * 512)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ic_chunk_inner = 0; ic_chunk_inner < 16; ++ic_chunk_inner) {
      #pragma unroll
      for (int oc_chunk = 0; oc_chunk < 2; ++oc_chunk) {
        #pragma unroll
        for (int oc_block = 0; oc_block < 4; ++oc_block) {
          compute[(((oc_chunk * 4) + oc_block))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[(((oc_chunk * 4) + oc_block))]);
          compute[((((oc_chunk * 4) + oc_block) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + (((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((oc_chunk * 4) + oc_block) + 16))]);
          compute[((((oc_chunk * 4) + oc_block) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)) + 224))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)))))[0], compute[((((oc_chunk * 4) + oc_block) + 8))]);
          compute[((((oc_chunk * 4) + oc_block) + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner * 448) + (((int)threadIdx.x) * 4)) + 224))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk * 256)) + (ic_chunk_inner * 16)) + (oc_block * 4)) + 2048))))[0], compute[((((oc_chunk * 4) + oc_block) + 24))]);
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) < 256) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 224) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) < 1024) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 4) + ((int)threadIdx.y)) < 19) {
            ((int*)((signed char*)placeholder_shared + ((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 896) + (((int)threadIdx.y) * 224)) + (((int)threadIdx.x) * 4)))))[0] = ((int*)((signed char*)placeholder1 + ((((((((int)blockIdx.y) * 8192) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) >> 4) * 512)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_outer_fused_outer_outer_outer1 * 56) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) >> 2)) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 256))))[0];
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ic_chunk_inner1 = 0; ic_chunk_inner1 < 16; ++ic_chunk_inner1) {
    #pragma unroll
    for (int oc_chunk1 = 0; oc_chunk1 < 2; ++oc_chunk1) {
      #pragma unroll
      for (int oc_block1 = 0; oc_block1 < 4; ++oc_block1) {
        compute[(((oc_chunk1 * 4) + oc_block1))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7168))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[(((oc_chunk1 * 4) + oc_block1))]);
        compute[((((oc_chunk1 * 4) + oc_block1) + 16))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7168))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute[((((oc_chunk1 * 4) + oc_block1) + 16))]);
        compute[((((oc_chunk1 * 4) + oc_block1) + 8))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7392))))[0], ((int*)((signed char*)placeholder_shared + (((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)))))[0], compute[((((oc_chunk1 * 4) + oc_block1) + 8))]);
        compute[((((oc_chunk1 * 4) + oc_block1) + 24))] = __dp4a(((int*)((signed char*)pad_data_shared + ((((ic_chunk_inner1 * 448) + (((int)threadIdx.x) * 4)) + 7392))))[0], ((int*)((signed char*)placeholder_shared + ((((((((int)threadIdx.y) * 512) + (oc_chunk1 * 256)) + (ic_chunk_inner1 * 16)) + (oc_block1 * 4)) + 2048))))[0], compute[((((oc_chunk1 * 4) + oc_block1) + 24))]);
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax4 = 0; ax4 < 4; ++ax4) {
      ((int*)T_relu)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147423160) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[(((ax1_inner_inner_inner * 4) + ax4))]) << ((long)18)) : ((long)compute[(((ax1_inner_inner_inner * 4) + ax4))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147423160) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1646512227) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4))]) << ((long)0)) : ((long)((int*)placeholder4)[((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4))])) * (long)1993989853) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2147423160) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 16))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2147423160) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)1646512227) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25088))])) * (long)1993989853) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 224))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147423160) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 8))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)2147423160) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[(((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4))]))) * (long)1646512227) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 224))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 224))])) * (long)1993989853) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
      ((int*)T_relu)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25312))] = max((int)((((int)(((((0 != 0) ? (((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2147423160) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((0 != 0) ? (((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))])) << ((long)0)) : ((long)(((int)(((((18 != 0) ? (((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))]) << ((long)18)) : ((long)compute[((((ax1_inner_inner_inner * 4) + ax4) + 24))])) * (long)1342381011) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder2)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)2147423160) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int*)placeholder3)[((((((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 8)) + (ax1_inner_inner_inner * 4)) + ax4) + 32))]))) * (long)1646512227) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))) + ((int)(((((0 != 0) ? (((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25312))]) << ((long)0)) : ((long)((int*)placeholder4)[(((((((((((int)blockIdx.z) * 401408) + (((int)blockIdx.y) * 50176)) + (((int)threadIdx.y) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.x) * 448)) + (((int)threadIdx.x) * 4)) + ax4) + 25312))])) * (long)1993989853) + ((long)1 << ((long)((0 + 31) - 1)))) >> ((long)(0 + 31)))))), (int)(0));
    }
  }
}

