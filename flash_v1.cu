#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/types.h>

// MHA
// Q, K, V, O shape (batch B, num_heads H, seq_len N, head_dim D)
__global__ void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int D, const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
    float* l, float* m, float* O) {
    int threadidx_x = threadIdx.x; // K/V tile row
    int threadidx_y = threadIdx.y;  // Q tile row

    int qkv_offset = blockIdx.x * gridDim.y * N * D + blockIdx.y * N * D;
    int lm_offset = blockIdx.x * gridDim.y * N + blockIdx.y * N;

    extern __shared__ float sram[];
    int tile_size_Qi = Br * D;
    int tile_size_KV_j = Bc * D;

    int offset_Kj = tile_size_Qi;
    int offset_Vj = offset_Kj + tile_size_KV_j;
    int offset_S = offset_Vj + tile_size_KV_j;

    float* Qi = sram;
    float* Kj = &sram[offset_Kj];
    float* Vj = &sram[offset_Vj];
    float* S  = &sram[offset_S];

    // outer loop K/V
   for (int j = 0; j < Tc; j++) {
        int valid_Bc = min(Bc, N - j * Bc);

        // load K/V
        if (threadidx_x < valid_Bc) {
            int kv_row = j * Bc + threadidx_x;
            #pragma unroll
            for (int x = 0; x < D; x++) {
                Kj[threadidx_x * D + x] = K[qkv_offset + kv_row * D + x];
                Vj[threadidx_x * D + x] = V[qkv_offset + kv_row * D + x];
            }
        }
        // inner loop Q
        for (int i = 0; i < Tr; i++) {
            int valid_Br = min(Br, N - i * Br);
            if (threadidx_y < valid_Br) {
                int q_row = i * Br + threadidx_y;
                #pragma unroll
                for (int x = 0; x < D; x++) {
                    Qi[threadidx_y * D + x] = Q[qkv_offset + q_row * D + x];
                }

                float row_m_prev = m[lm_offset + q_row];
                float row_l_prev = l[lm_offset + q_row];

                float row_m = -INFINITY;
                for (int y = 0; y < valid_Bc; y++) {
                    float sum = 0;
                    for (int x = 0; x < D; x++) {
                        sum += Qi[threadidx_y * D + x] * Kj[y * D + x];
                    }
                    sum *= softmax_scale;
                    S[threadidx_y * Bc + y] = sum;
                    if (sum > row_m) row_m = sum;
                }

                float row_l = 0;
                #pragma unroll
                for (int y = 0; y < valid_Bc; y++) {
                    S[threadidx_y * Bc + y] = __expf(S[threadidx_y * Bc + y] - row_m);
                    row_l += S[threadidx_y * Bc + y];
                }

                float row_m_new = max(row_m_prev, row_m);
                float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

                for (int x = 0; x < D; x++) {
                    float pv = 0;
                    #pragma unroll
                    for (int y = 0; y < valid_Bc; y++) {
                        pv += S[threadidx_y * Bc + y] * Vj[y * D + x];
                    }
                    int o_offset = qkv_offset + q_row * D + x;
                    O[o_offset] = (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[o_offset]) + (__expf(row_m - row_m_new) * pv));
                }

                m[lm_offset + q_row] = row_m_new;
                l[lm_offset + q_row] = row_l_new;
            }
            __syncthreads();
        } 
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int Bc = 16;
    const int Br = 32;
    const int B  = Q.size(0);
    const int nH = Q.size(1);
    const int N  = Q.size(2);
    const int D  = Q.size(3);

    const int Tc = ceil((float) N / Bc);
    const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1. / sqrt(D);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nH, N}, Q.options());
    auto m = torch::full({B, nH, N}, -INFINITY, Q.options());

    const int sram_size = (Br * D + 2 * Bc * D + Br * Bc) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

    dim3 grid_dim(B, nH);
    dim3 block_dim(Bc, Br);
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, D, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );

    return O;
}
