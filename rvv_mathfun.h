#ifndef RVV_MATHFUN_H
#define RVV_MATHFUN_H
#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#endif

#define c_inv_mant_mask ~0x7f800000u
#define c_cephes_SQRTHF 0.707106781186547524
#define c_cephes_log_p0 7.0376836292E-2
#define c_cephes_log_p1 -1.1514610310E-1
#define c_cephes_log_p2 1.1676998740E-1
#define c_cephes_log_p3 -1.2420140846E-1
#define c_cephes_log_p4 +1.4249322787E-1
#define c_cephes_log_p5 -1.6668057665E-1
#define c_cephes_log_p6 +2.0000714765E-1
#define c_cephes_log_p7 -2.4999993993E-1
#define c_cephes_log_p8 +3.3333331174E-1
#define c_cephes_log_q1 -2.12194440e-4
#define c_cephes_log_q2 0.693359375

#define _RVV_FLOAT32_LOG_OP(LMUL, MLEN)                                                                              \
    static inline vfloat32m##LMUL##_t log_ps(vfloat32m##LMUL##_t x, size_t vl)                                       \
    {                                                                                                                \
        x = __riscv_vfmax_vf_f32m##LMUL(x, 0.f, vl); /* force flush to zero on denormal values */                            \
        vbool##MLEN##_t invalid_mask = __riscv_vmfle_vf_f32m##LMUL##_b##MLEN(x, 0.f, vl);                                    \
                                                                                                                     \
        vint32m##LMUL##_t ux = __riscv_vreinterpret_v_f32m##LMUL##_i32m##LMUL(x);                                            \
                                                                                                                     \
        vint32m##LMUL##_t emm0 = __riscv_vsra_vx_i32m##LMUL(ux, 23, vl);                                                     \
                                                                                                                     \
        /* keep only the fractional part */                                                                          \
        ux = __riscv_vand_vx_i32m##LMUL(ux, c_inv_mant_mask, vl);                                                            \
        ux = __riscv_vor_vx_i32m##LMUL(ux, 1056964608 /* reinterpret_cast<int>(0.5) */, vl);                                 \
        x = __riscv_vreinterpret_v_i32m##LMUL##_f32m##LMUL(ux);                                                              \
                                                                                                                     \
        emm0 = __riscv_vsub_vx_i32m##LMUL(emm0, 0x7f, vl);                                                                   \
        vfloat32m##LMUL##_t e = __riscv_vfcvt_f_x_v_f32m##LMUL(emm0, vl);                                                    \
                                                                                                                     \
        e = __riscv_vfadd_vf_f32m##LMUL(e, 1.f, vl);                                                                         \
                                                                                                                     \
        /* part2:                      */                                                                            \
        /*     if( x < SQRTHF ) {      */                                                                            \
        /*       e -= 1;               */                                                                            \
        /*       x = x + x - 1.0;      */                                                                            \
        /*     } else { x = x - 1.0; } */                                                                            \
        vbool##MLEN##_t mask = __riscv_vmflt_vf_f32m##LMUL##_b##MLEN(x, c_cephes_SQRTHF, vl);                                \
        x = __riscv_vfadd_vv_f32m##LMUL##_mu(mask, x, x, x, vl);                                                              \
        x = __riscv_vfsub_vf_f32m##LMUL(x, 1.f, vl);                                                                         \
        e = __riscv_vfsub_vf_f32m##LMUL##_mu(mask, e, e, 1.f, vl);                                                            \
                                                                                                                     \
        vfloat32m##LMUL##_t z = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);                                                       \
                                                                                                                     \
        vfloat32m##LMUL##_t y = __riscv_vfmul_vf_f32m##LMUL(x, c_cephes_log_p0, vl);                                         \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p1, vl);                                                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                           \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p2, vl);                                                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                           \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p3, vl);                                                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                           \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p4, vl);                                                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                           \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p5, vl);                                                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                           \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p6, vl);                                                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                           \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p7, vl);                                                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                           \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_log_p8, vl);                                                             \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                                           \
                                                                                                                     \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                                                           \
                                                                                                                     \
        vfloat32m##LMUL##_t tmp = __riscv_vfmul_vf_f32m##LMUL(e, c_cephes_log_q1, vl);                                       \
        y = __riscv_vfadd_vv_f32m##LMUL(y, tmp, vl);                                                                         \
                                                                                                                     \
        tmp = __riscv_vfmul_vf_f32m##LMUL(z, 0.5f, vl);                                                                      \
        y = __riscv_vfsub_vv_f32m##LMUL(y, tmp, vl);                                                                         \
                                                                                                                     \
        tmp = __riscv_vfmul_vf_f32m##LMUL(e, c_cephes_log_q2, vl);                                                           \
        x = __riscv_vfadd_vv_f32m##LMUL(x, y, vl);                                                                           \
        x = __riscv_vfadd_vv_f32m##LMUL(x, tmp, vl);                                                                         \
        /* negative arg will be NAN */                                                                               \
        vuint32m##LMUL##_t xtmp = __riscv_vreinterpret_v_f32m##LMUL##_u32m##LMUL(x);                                         \
        x = __riscv_vreinterpret_v_u32m##LMUL##_f32m##LMUL(__riscv_vor_vx_u32m##LMUL##_mu(invalid_mask, xtmp, xtmp, 0xffffffff, vl)); \
        return x;                                                                                                    \
    }

_RVV_FLOAT32_LOG_OP(8, 4)

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

#define _RVV_FLOAT32_EXP_OP(LMUL, MLEN)                                                   \
    static inline vfloat32m##LMUL##_t exp_ps(vfloat32m##LMUL##_t x, size_t vl)            \
    {                                                                                     \
        vfloat32m##LMUL##_t tmp, fx;                                                      \
                                                                                          \
        x = __riscv_vfmin_vf_f32m##LMUL(x, c_exp_hi, vl);                                         \
        x = __riscv_vfmax_vf_f32m##LMUL(x, c_exp_lo, vl);                                         \
                                                                                          \
        /* express exp(x) as exp(g + n*log(2)) */                                         \
        fx = __riscv_vfmacc_vf_f32m##LMUL##_tu( __riscv_vfmv_v_f_f32m##LMUL(0.5f, vl), c_cephes_LOG2EF, x, vl); \
                                                                                          \
        /* perform a floorf */                                                            \
        tmp = __riscv_vfcvt_f_x_v_f32m##LMUL(__riscv_vfcvt_x_f_v_i32m##LMUL(fx, vl), vl);                 \
                                                                                          \
        /* if greater, substract 1 */                                                     \
        vbool##MLEN##_t mask = __riscv_vmfgt_vv_f32m##LMUL##_b##MLEN(tmp, fx, vl);                \
        fx = __riscv_vfsub_vf_f32m##LMUL##_mu(mask, tmp, tmp, 1.f, vl);                            \
                                                                                          \
        tmp = __riscv_vfmul_vf_f32m##LMUL(fx, c_cephes_exp_C1, vl);                               \
        vfloat32m##LMUL##_t z = __riscv_vfmul_vf_f32m##LMUL(fx, c_cephes_exp_C2, vl);             \
        x = __riscv_vfsub_vv_f32m##LMUL(x, tmp, vl);                                              \
        x = __riscv_vfsub_vv_f32m##LMUL(x, z, vl);                                                \
                                                                                          \
        vfloat32m##LMUL##_t y = __riscv_vfmul_vf_f32m##LMUL(x, c_cephes_exp_p0, vl);              \
        z = __riscv_vfmul_vv_f32m##LMUL(x, x, vl);                                                \
                                                                                          \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p1, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p2, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p3, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p4, vl);                                  \
        y = __riscv_vfmul_vv_f32m##LMUL(y, x, vl);                                                \
        y = __riscv_vfadd_vf_f32m##LMUL(y, c_cephes_exp_p5, vl);                                  \
                                                                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, z, vl);                                                \
        y = __riscv_vfadd_vv_f32m##LMUL(y, x, vl);                                                \
        y = __riscv_vfadd_vf_f32m##LMUL(y, 1.f, vl);                                              \
                                                                                          \
        /* build 2^n */                                                                   \
        vint32m##LMUL##_t mm = __riscv_vfcvt_x_f_v_i32m##LMUL(fx, vl);                            \
        mm = __riscv_vadd_vx_i32m##LMUL(mm, 0x7f, vl);                                            \
        mm = __riscv_vsll_vx_i32m##LMUL(mm, 23, vl);                                              \
        vfloat32m##LMUL##_t pow2n = __riscv_vreinterpret_v_i32m##LMUL##_f32m##LMUL(mm);           \
                                                                                          \
        y = __riscv_vfmul_vv_f32m##LMUL(y, pow2n, vl);                                            \
        return y;                                                                         \
    }

_RVV_FLOAT32_EXP_OP(8, 4)


#endif
