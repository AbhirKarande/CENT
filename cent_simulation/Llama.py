import math
import torch
import torch.nn.functional as F
from aim_sim import PIM
from TransformerBlock import TransformerBlock
from utils import compare, apply_rotary_emb, repeat_kv, RMSNorm

debug = True

class TransformerBlockLlama(TransformerBlock):
    """
    TransformerBlock Class inherits computate functionality from PIM class
    """
    def __init__(self, dic_model, args):
        super().__init__(dic_model, args)
        
    def precision_test(self):
        # Results are different in BFloat16 in 7B
        a = RMSNorm(self.x, self.SANorm)[0][0]
        b = self.wq[34]
        c = self.wq.T[:, 34:35]
        print(a)
        print(b)
        print(c)

        print(self.Vector_Vector_Mul(a, b, False))
        print((a*b).sum())
        print(torch.matmul(a, b))
        print(torch.matmul(a, c))

    def self_attention(self):
        bsz, seqlen, _ = self.x.shape

        RMSNorm_x = RMSNorm(self.x, self.SANorm)

        xq = F.linear(RMSNorm_x, self.wq)
        xk = F.linear(RMSNorm_x, self.wk)
        xv = F.linear(RMSNorm_x, self.wv)
        compare(xq[0][0], self.xq[0][0], "xq")
        compare(xk[0][0], self.xk[0][0], "xk")
        compare(xv[0][0], self.xv[0][0], "xv")
        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis)
        self.cache_k[:bsz, self.start_pos : self.start_pos + seqlen] = xk
        self.cache_v[:bsz, self.start_pos : self.start_pos + seqlen] = xv
        keys = self.cache_k[:bsz, : self.start_pos + seqlen]
        values = self.cache_v[:bsz, : self.start_pos + seqlen]
        if self.GQA:
            keys = repeat_kv(keys, self.n_repeat)
            values = repeat_kv(values, self.n_repeat)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2).transpose(2, 3)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys) / math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1).type_as(xq)
        compare(scores[0][0], self.scores[0][0], "scores")

        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)
        compare(output[0][0], self.output[0][0], "output")
        sa = F.linear(output, self.wo)
        compare(sa[0][0], self.sa[0][0], "sa")
        sa = self.x + sa
        return sa
    
    def self_attention_aim(self):
        #unpack shape of input tensor (self.x). bsz is the batch size.
        bsz, _, _ = self.x.shape
        #sequence length is calculated as the current length of the input sequence being processed. 
        #self.start_pos is the starting position of the current sequence in the input tensor.
        seqlen = self.start_pos + 1
        #if model parallelism is enabled, the total number of banks is scaled by the number of FC devices. i.e. the total number of memory banks available for the fully connected layers is the number of banks per device multiplied by the number of devices
        #channels required is the number of channels required for the current transformer block.
        if self.model_parallel:
            FC_total_banks = self.total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            #if model is running on single device, the number of banks is th total available on the single device
            FC_total_banks = self.total_banks
            #The computation for this specific transformer block will only use a subset of the total channels available on the device. 
            #This allows for other transformer blocks to run in parallel on the same device using different channels (pipeline parallelism)
            channels_required = self.channels_per_block
            #Simple list of integers representing the indices of memory channels that this specific transformer block will use for its computations
        channel_lst = [channel for channel in range(channels_required)]
        #calculates the total number of channels on the device that can be evenly divided among transformer blocks. For example if a device has 32 channels and each block needs 8, this value is 32. If each block needs 10, this value is 30
        channel_multi_transformer_block_required = self.num_channels // channels_required * channels_required
        channel_lst_multi_transformer_block = [channel for channel in range(channel_multi_transformer_block_required)]

        # AiM MAC BK x BK
        #if pim_compute is true, the computation is performed on the PIM. 

        if self.pim_compute:
            #RMS Normalization
            #initializes the variable to hold the final sum
            x_pow_sum = 0
            #calculates the number of PIM operations required to process the entire input vector, based on its size and the hardware's burst length.  How to define burst length???
            op_size = (self.dic_shape["x_neighbor_bank"][0] - 1) // self.burst_length + 1
            #loops through each memory channel to assign to this transformer block
            for channel in channel_lst:
                #flag to control whether detailed performance tracing is logged. It's only turned on for the first channel to avoid redundant logs
                op_trace = channel == 0 and self.trace_norm
                #Simulates writing a bias value of 0 to the hardware's Multiply-Accumulate (MAC) unit. This effectively resets them before the calculation begins.
                self.WR_BIAS(0, channel, channels_required, 0, [0 for bank in range(self.num_banks)], op_trace)
                #Multiply-Accumulate, Bank-to-Bank. Simulates an element wise multiplication of the input vector (x, stored in self.x_row_index) with itself and accumlates the results. (equivalent to sum(x*x))
                self.MAC_BK_BK(0, channel, channels_required, self.x_row_index, 0, 0, op_size, op_trace)
                #after the in-memory computation, this simulates reading the accumulated results from MAC units for that channel
                mac_lst = self.RD_MAC(0, channel, channels_required, 0, op_trace)
                #partial sum from the channel is added to the total. The comment CXL ports indicates this data transfer is being simulated over the CXL bus.
                x_pow_sum += sum(mac_lst)    # CXL ports
                #It compares the results from the PIM simulation with the result calculated using standard PyTorch (self.x.pow(2.sum()) to ensure the simulation is numerically correct)
            compare(x_pow_sum, self.x.pow(2).sum(), "x_pow_sum")
        #Else condition is assuming that we are simulating DRAM data loads followed by CPU calculation
        else:
            #simulates loading the input vector (x) from the modeled DRAM.
            x_load = self.load_from_DRAM_multi_channel(self.x.shape, self.x_row_index, "vector_neighbor_bank_0", self.dic_shape["x_neighbor_bank"][0], False)
            #Simulates loading a second copy of the same vector. This is done to mimic the hardware operation which takes two operands
            x_copy_load = self.load_from_DRAM_multi_channel(self.x.shape, self.x_row_index, "vector_neighbor_bank_1", self.dic_shape["x_neighbor_bank"][0], False)
            #Calls a custom CPU-based function to perform an element-wise multiplication of the two loaded vectors and sum the result, giving the sum of squares
            x_pow_sum = self.Vector_Vector_Mul(x_load[0][0], x_copy_load[0][0], False)

        # CXL Ports     x_copy -> norm_tensor, This operation involves a data transfer over CXL bus.
        #RMSNorm step to compute the mean of the squares by dividing x_pow_sum by the vector dimension (self.dim) and adding a small epsilon to avoid division by zero.
        #calculate the reciprocal of the square root of that value. THe result, norm, is the single scalar value that will be used to normalize the entire input vector
        norm = torch.rsqrt(x_pow_sum / self.dim + 1e-5)
        #creates new tensor, norm_tensor, that has the exact same shape as the original input self.x. 
        #the entire tensor is filled with the scalar norm value. This effectively "broadcasts" the normalization factor so that it can be used for an element-wise multiplication with the input vector in the next stage. 
        norm_tensor = torch.full(self.x.shape, norm)
        #Simulates writing norm_tensor to the model's DRAM. It logs the details of this memory operation, including the destination address (self.x_copy_row_index) and the memory layout ("vector_bank_group_1"). Helps in modeling bandwidth usage
        self.store_to_DRAM_multi_channel(norm_tensor[0][0], self.x_copy_row_index, "vector_bank_group_1", self.trace_norm)
        #updates the simulation's performance counters. It adds the time taken for thiw write operation to the total time.
        #Time is calculated as a fixed latency (self.timing_constant["WR_SBK"]) plus the number of elements in the vector divided by the burst length. (self.x.shape[-1] is the number of elements in the vector)
        #(self.x.shape[-1]//self.burst_length) is the number of PIM operations required to write the entire vector to DRAM. 
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
        
        # AiM EWMUL     Load x and norm_tensor      norm_tensor -> norm_x -> RMSNorm_x_aim
        
        if self.pim_compute:
            #calculates number of hardware operations required based on the data size and the PIM device's burst length
            op_size = (self.dic_shape["x_bank_group"][0] - 1) // self.burst_length + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                #Simulates hardware-level Element Wise Multiplication instruction. input (x) in vector_bank_group_0 and norm_tensor in vector_bank_group_1
                self.EWMUL(0, channel, channels_required, self.x_copy_row_index, 0, op_size, op_trace)
                #result is implicitly stored in vector_bank_group_2
        else:
            #simulates loading norm_tensor from DRAM into the CPU
            norm_tensor_load = self.load_from_DRAM_multi_channel(self.x.shape, self.x_copy_row_index, "vector_bank_group_1", self.dic_shape["x_bank_group"][0], False)
            #Simulates loading the input tensor (x) from DRAM into the CPU
            x_load = self.load_from_DRAM_multi_channel(self.x.shape, self.x_copy_row_index, "vector_bank_group_0", self.dic_shape["x_bank_group"][0], False)
            #Performs element-wise multiplication on CPU using the two loaded tensors
            norm_x = self.Vector_Vector_EWMUL(x_load, norm_tensor_load)
            #Simulates writing the resulting normalized tensor (norm_x) to DRAM)
            self.store_to_DRAM_multi_channel(norm_x[0][0], self.x_copy_row_index, "vector_bank_group_2", False)

        # AiM EWMUL     Copy norm_x to SANorm row
        if self.pim_compute:
            op_size = (self.dic_shape["x_bank_group"][0] - 1) // self.burst_length + 1
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_norm
                    self.COPY_BK_GB(0, channel, channels_required, bank, self.x_copy_row_index, 0, op_size, op_trace)
                    self.COPY_GB_BK(0, channel, channels_required, bank-1, self.SANorm_row_index, 0, op_size, op_trace)
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.SANorm_row_index, 0, op_size, op_trace)
        else:
            self.store_to_DRAM_multi_channel(norm_x[0][0], self.SANorm_row_index, "vector_bank_group_1", False)
            norm_x_load = self.load_from_DRAM_multi_channel(self.x.shape, self.SANorm_row_index, "vector_bank_group_1", self.dic_shape["x_bank_group"][0], False)
            SANorm_load = self.load_from_DRAM_multi_channel(self.x.shape, self.SANorm_row_index, "vector_bank_group_0", self.dic_shape["SANorm_bank_group"][0], False)
            RMSNorm_x_aim = self.Vector_Vector_EWMUL(norm_x_load, SANorm_load)
            self.store_to_DRAM_multi_channel(RMSNorm_x_aim[0][0], self.SANorm_row_index, "vector_bank_group_2", False)
        
        # Broadcast the scattered RMSNorm_x_aim results to all channels
        RMSNorm_x_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.SANorm_row_index, "vector_bank_group_2", self.dic_shape["x_bank_group"][0], self.trace_norm)
        compare(RMSNorm_x_aim[0][0], RMSNorm(self.x[0][0], self.SANorm), "RMSNorm_x_aim")
        
        # AiM MAC BK x GB
        #!!!!!Q, K, V projections!!!!!
        if self.pim_compute:
        # if False:
            xq_aim = self.Vector_Matrix_Mul_weight_pim(RMSNorm_x_aim[0][0], self.wq_row_index, self.dim, self.wq.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
            xk_aim = self.Vector_Matrix_Mul_weight_pim(RMSNorm_x_aim[0][0], self.wk_row_index, self.dim, self.wk.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
            xv_aim = self.Vector_Matrix_Mul_weight_pim(RMSNorm_x_aim[0][0], self.wv_row_index, self.dim, self.wv.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
        else:
            wq_aim = self.load_from_DRAM_multi_channel(self.wq.shape, self.wq_row_index, self.mode["weights"], self.dic_shape["wq"][0], False)
            wk_aim = self.load_from_DRAM_multi_channel(self.wk.shape, self.wk_row_index, self.mode["weights"], self.dic_shape["wk"][0], False)
            wv_aim = self.load_from_DRAM_multi_channel(self.wv.shape, self.wv_row_index, self.mode["weights"], self.dic_shape["wv"][0], False)
            xq_aim = self.Vector_Matrix_Mul_multithreads(RMSNorm_x_aim[0][0], wq_aim.T).reshape(bsz, 1, -1)
            xk_aim = self.Vector_Matrix_Mul_multithreads(RMSNorm_x_aim[0][0], wk_aim.T).reshape(bsz, 1, -1)
            xv_aim = self.Vector_Matrix_Mul_multithreads(RMSNorm_x_aim[0][0], wv_aim.T).reshape(bsz, 1, -1)
            # xq_aim = torch.tensor(self.Vector_Matrix_Mul(RMSNorm_x_aim[0][0], wq_aim.T)).reshape(bsz, 1, -1)
            # xk_aim = torch.tensor(self.Vector_Matrix_Mul(RMSNorm_x_aim[0][0], wk_aim.T)).reshape(bsz, 1, -1)
            # xv_aim = torch.tensor(self.Vector_Matrix_Mul(RMSNorm_x_aim[0][0], wv_aim.T)).reshape(bsz, 1, -1)
        compare(xq_aim[0][0], self.xq[0][0], "Vector_Matrix_Mul xq")
        compare(xk_aim[0][0], self.xk[0][0], "Vector_Matrix_Mul xk")
        compare(xv_aim[0][0], self.xv[0][0], "Vector_Matrix_Mul xv")

        # CXL Ports     rotary embedding
        xq_aim = xq_aim.reshape(bsz, 1, self.n_heads, self.head_dim)
        xk_aim = xk_aim.reshape(bsz, 1, self.n_kv_heads, self.head_dim)
        xv_aim = xv_aim.reshape(bsz, 1, self.n_kv_heads, self.head_dim)

        xq_aim, xk_aim = apply_rotary_emb(xq_aim, xk_aim, self.freqs_cis)

        if self.trace_fc_kqvo:
            input_vector_EWMUL_length = (self.dim - 1) // (self.total_banks // 4) + 1
            input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1
            # Store re-mapped xq/xk for EWMUL
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            # Rotary embedding
            self.EWMUL_only_trace(channel_lst_multi_transformer_block, self.xq_row_index, self.dim // self.burst_length)
            self.EWMUL_only_trace(channel_lst_multi_transformer_block, self.xk_row_index, self.dim // self.n_repeat // self.burst_length)
            # Load rotary embedding results
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)

        self.dic_shape["xq"] = self.store_to_DRAM_multi_channel(xq_aim.reshape(-1), self.xq_row_index, self.mode["vector"], False)

        # CXL Ports     Store xq
        if self.pim_compute:
            self.broadcast_store_query(channels_required, self.xq_row_index, xq_aim.reshape(-1), False)
            xq_aim_loaded = {}
            self.broadcast_load_query(xq_aim_loaded, channels_required, self.xq_row_index)
            print()
            for channel in channel_lst:
                compare(xq_aim.reshape(-1), xq_aim_loaded[channel], "xq_aim_loaded channel "+str(channel))
        else:
            xq_aim_load = self.load_from_DRAM_multi_channel(self.xq.shape, self.xq_row_index, self.mode["vector"], self.dic_shape["xq"][0], False)
            xq_aim_load = xq_aim_load.reshape(bsz, 1, self.n_heads, self.head_dim)
            compare(xq_aim_load[0][0], xq_aim[0][0], "xq_aim_load")

        # CXL Ports     Store xk xv
        if self.pim_compute:
            # cache k and v in weights have reserved the positions for xk and xv, for each processed token, we need to store the new xk/xv to the correct position
            seq = seqlen - 1
            dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
            xk_data = xk_aim.reshape(-1)
            rows = self.head_dim * self.n_kv_heads // self.DRAM_column
            for row in range(rows):
                data_row = xk_data[row * self.DRAM_column : (row + 1) * self.DRAM_column]
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.DRAM_column // self.burst_length
                self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, self.cache_k_row_index + seq // self.FC_total_banks * rows + row, 0, self.DRAM_column, data_row, self.trace_attention)

            if self.intra_device_attention:   # old V cache mapping, which is not friendly for long context scenario. E.g. seqlen=32k, it stores 32 rows data (1k per row) in each bank, and utilizes 128 banks (8 channels) per head. In Llama2-70B with GQA, only 2 devices (16x32x2=1k banks) can be used. 
                xv_data = xv_aim.transpose(1, 2).transpose(2, 3)
                num_rows_per_seq = (seq - 1) // self.DRAM_column + 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                dim_iteration = self.head_dim // self.num_banks
                for head_index_per_bank in range(num_heads_per_bank):     # each head is distributed into all banks in a channel, each bank contains left_banks heads
                    row_current_head = self.cache_v_row_index + (rows_per_dim * dim_iteration) * head_index_per_bank
                    for dim_iter in range(dim_iteration):   # each head has dim 128, but distributed to 16 banks, so has 8 iterations in each bank
                        for channel in channel_lst:
                            head = channel * num_heads_per_bank + head_index_per_bank
                            if head > self.n_kv_heads - 1:
                                break
                            if self.trace_attention:
                                self.WR_ABK_only_trace(channel, row_current_head + dim_iter * rows_per_dim + num_rows_per_seq - 1, 1)
                            # self.store_to_DRAM_all_banks(dim_iter, channel, row_current_head, seq, head, xv_data, num_rows_per_seq, rows_per_dim)
                            for bank in range(self.num_banks):
                                dim = dim_iter * self.num_banks + bank
                                row_offset = num_rows_per_seq - 1
                                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + 1
                                self.store_to_DRAM_single_bank(0, channel, bank, row_current_head + dim_iter * rows_per_dim + row_offset, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)
            else:
                xv_data = xv_aim.transpose(1, 2).transpose(2, 3)
                channels_required_all_devices = self.FC_total_banks // self.num_banks
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                for channel in range(channels_required_all_devices):
                    if banks_per_head < self.num_banks:
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            row_offset = rows_per_seq - 1
                            if self.trace_attention and channel < self.num_channels:
                                self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, 1)
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel % channels_per_head) * self.num_banks + bank
                                self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            if (channel % channels_per_head) // channels_per_row_offset == rows_per_seq - 1:
                                row_offset = rows_per_seq - 1
                                if self.trace_attention and channel < self.num_channels:
                                    self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset // rows_per_seq_iteration, 1)
                                for bank in range(self.num_banks):
                                    dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                                    self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset // rows_per_seq_iteration, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)

        else:
            cache_v_size = torch.Size([bsz, self.n_kv_heads, self.head_dim, -1])
            cache_k = self.load_from_DRAM_multi_channel(self.cache_k.shape, self.cache_k_row_index, self.mode["cache_k"], self.cache_k.shape[1], False)
            cache_v = self.load_from_DRAM_multi_channel(cache_v_size, self.cache_v_row_index, self.mode["cache_v"], self.cache_k.shape[1], False).transpose(2, 3).transpose(1, 2).reshape(bsz, -1, self.n_kv_heads, self.head_dim)
            compare(cache_k, self.cache_k, "cache v old")
            compare(cache_v, self.cache_v, "cache k old")
            cache_k[:bsz, self.start_pos : self.start_pos + 1] = xk_aim
            cache_v[:bsz, self.start_pos : self.start_pos + 1] = xv_aim

            keys_aim = cache_k[:bsz, : self.start_pos + 1]
            values_aim = cache_v[:bsz, : self.start_pos + 1]
            if self.GQA:
                keys_aim = repeat_kv(keys_aim, self.n_repeat)
                values_aim = repeat_kv(values_aim, self.n_repeat)
            xq_aim_load = xq_aim_load.transpose(1, 2)
            keys_aim = keys_aim.transpose(1, 2).transpose(2, 3)
            values_aim = values_aim.transpose(1, 2)

        # AiM MAC BK x GB
        if self.pim_compute:
            scores_aim = self.Vector_Matrix_Mul_score_pim(self.xq_row_index, self.cache_k_row_index, self.trace_attention, "breakdown_sa_score")

            if debug:
                self.cache_k[:bsz, self.start_pos : self.start_pos + seqlen] = xk_aim
                self.cache_v[:bsz, self.start_pos : self.start_pos + seqlen] = xv_aim
                keys = self.cache_k[:bsz, : self.start_pos + seqlen]
                values = self.cache_v[:bsz, : self.start_pos + seqlen]
                if self.GQA:
                    keys = repeat_kv(keys, self.n_repeat)
                    values = repeat_kv(values, self.n_repeat)
                xq = xq_aim.transpose(1, 2)
                keys = keys.transpose(1, 2).transpose(2, 3)
                compare(scores_aim, torch.matmul(xq, keys), "Vector_Matrix_Mul score")
        else:
            scores_aim = []
            for i in range(self.n_heads):
                scores_aim.append(self.Vector_Matrix_Mul(xq_aim_load[0][i][0], keys_aim[0][i], False))
            scores_aim = torch.tensor(scores_aim).reshape(bsz, self.n_heads, 1, -1)

        # CXL Ports
        head_dim_reciprocal = torch.full(scores_aim.shape, 1 / math.sqrt(self.head_dim))
        self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(head_dim_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in channel_lst:
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
        else:
            scores_aim_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            head_dim_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_aim_load, head_dim_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)

        # CXL Ports
        scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        if self.pim_compute and debug:
            scores = torch.matmul(xq, keys) / math.sqrt(self.head_dim)
            compare(scores_aim, scores, "Vector_Matrix_Mul score / head_dim")

        scores_exp = torch.exp(scores_aim)
        scores_exp_sum_reciprocal = 1 / torch.sum(scores_exp, dim=-1, keepdim=True)
        scores_exp_sum_reciprocal = torch.cat([scores_exp_sum_reciprocal for i in range(scores_exp.shape[-1])], dim=-1)
        self.store_to_DRAM_multi_channel(scores_exp.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(scores_exp_sum_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in range(channels_required):
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
            scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        else:
            scores_exp_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            scores_exp_sum_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_exp_load, scores_exp_sum_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)
        compare(scores_aim, self.scores, "SoftMax scores")

        # AiM MAC BK x GB
        if self.pim_compute:
            output_aim = self.Vector_Matrix_Mul_output_pim(scores_aim, self.cache_v_row_index, self.trace_attention, "breakdown_sa_output").reshape(bsz, 1, -1)
        else:
            output_aim = []
            for i in range(self.n_heads):
                output_aim.append(self.Vector_Matrix_Mul(scores_aim[0][i][0], values_aim[0][i], False))
            output_aim = torch.tensor(output_aim).reshape(bsz, 1, -1)
        compare(output_aim[0][0], self.output[0][0], "Vector_Matrix_Mul output")

        # CXL Ports
        self.dic_shape["output"] = self.store_to_DRAM_multi_channel(output_aim[0][0], self.output_row_index, self.mode["vector"], False)

        # AiM MAC BK x GB
        if self.pim_compute:
            sa_aim = self.Vector_Matrix_Mul_weight_pim(output_aim[0][0], self.wo_row_index, self.dim, self.wo.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
        else:
            wo_aim = self.load_from_DRAM_multi_channel(self.wo.shape, self.wo_row_index, self.mode["weights"], self.dic_shape["wo"][0], False)
            sa_aim = self.Vector_Matrix_Mul_multithreads(output_aim[0][0], wo_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_0", False)
        compare(sa_aim[0][0], self.sa[0][0], "Vector_Matrix_Mul sa")

        # CXL Ports
        sa_aim_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
        x_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
        sa_aim = self.Vector_Vector_EWADD(x_load, sa_aim_load)
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_2", False)

        return sa_aim
    
    def FFN(self, sa):
        compare(sa[0][0], self.h[0][0], "h")
        RMSNorm_sa = RMSNorm(sa, self.FFNNorm)
        x1 = F.linear(RMSNorm_sa, self.w1)
        x3 = F.linear(RMSNorm_sa, self.w3)
        ffn = F.linear(F.silu(x1) * x3, self.w2)
        compare(ffn[0][0], self.ffn[0][0], "ffn")
        out = sa + ffn
        return out
    
    def FFN_aim(self, sa_aim):
        bsz, _, _ = self.sa.shape
        if self.model_parallel:
            FC_total_banks = self.total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = self.total_banks
            channels_required = self.channels_per_block
        channel_lst = [channel for channel in range(channels_required)]

        # AiM MAC BK x BK
        if self.pim_compute:
            self.dic_shape["sa_neighbor_bank"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_0", self.trace_norm)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_1", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            sa_pow_sum = 0
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.WR_BIAS(0, channel, channels_required, 0, [0 for bank in range(self.num_banks)], op_trace)
                op_size = (self.dic_shape["sa_neighbor_bank"][0] - 1) // self.burst_length + 1
                self.MAC_BK_BK(0, channel, channels_required, self.sa_copy_row_index, 0, 0, op_size, op_trace)
                sa_pow_sum += sum(self.RD_MAC(0, channel, channels_required, 0, op_trace))    # CXL ports
        else:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", False)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_1", False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            sa_copy_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_pow_sum = self.Vector_Vector_Mul(sa_load[0][0], sa_copy_load[0][0], False)

        # CXL Ports
        compare(sa_pow_sum, sa_aim.pow(2).sum(), "sa pow")
        norm = torch.rsqrt(sa_pow_sum / self.dim + 1e-5)
        norm_tensor = torch.full(sa_aim.shape, norm)
        self.store_to_DRAM_multi_channel(norm_tensor[0][0], self.sa_copy_row_index, "vector_bank_group_1", self.trace_norm)
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length

        # AiM EWMUL
        if self.pim_compute:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.sa_copy_row_index, 0, op_size, op_trace)
        else:
            norm_tensor_load = self.load_from_DRAM_multi_channel(self.x.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            norm_sa = self.Vector_Vector_EWMUL(sa_load, norm_tensor_load)
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        # AiM EWMUL
        if self.pim_compute:
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_norm
                    self.COPY_BK_GB(0, channel, channels_required, bank, self.sa_copy_row_index, 0, op_size, op_trace)
                    self.COPY_GB_BK(0, channel, channels_required, bank-1, self.FFNNorm_row_index, 0, op_size, op_trace)
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.FFNNorm_row_index, 0, op_size, op_trace)
            FFNNorm_sa_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_2", self.dic_shape["FFNNorm"][0], self.trace_norm)
        else:
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_1", False)
            norm_sa_load = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            FFNNorm_load = self.load_from_DRAM_multi_channel(self.FFNNorm.shape, self.FFNNorm_row_index, "vector_bank_group_0", self.dic_shape["FFNNorm"][0], False)
            FFNNorm_sa_aim = self.Vector_Vector_EWMUL(norm_sa_load, FFNNorm_load)
            self.store_to_DRAM_multi_channel(FFNNorm_sa_aim[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        bsz, _, _ = FFNNorm_sa_aim.shape
        compare(FFNNorm_sa_aim[0][0], RMSNorm(sa_aim[0][0], self.FFNNorm), "FFNNorm_sa_aim")

        # AiM MAC BK x GB
        if self.pim_compute:
            x1_aim, x1_sigmoid_aim = self.Vector_Matrix_Mul_weight_af_pim(FFNNorm_sa_aim[0][0], self.w1_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")
            x1_aim = x1_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x1_sigmoid_aim = x1_sigmoid_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_weight_pim(FFNNorm_sa_aim[0][0], self.w3_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")[:self.w3.shape[0]].reshape(bsz, 1, -1)
        else:
            w1_aim = self.load_from_DRAM_multi_channel(self.w1.shape, self.w1_row_index, self.mode["weights"], self.dic_shape["w1"][0], False)
            w3_aim = self.load_from_DRAM_multi_channel(self.w3.shape, self.w3_row_index, self.mode["weights"], self.dic_shape["w3"][0], False)
            x1_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w1_aim.T).reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w3_aim.T).reshape(bsz, 1, -1)
            self.dic_shape["x1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_row_index, self.mode["vector"], False)
            self.dic_shape["x3"] = self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x3_row_index, self.mode["vector"], False)
        compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
        compare(x3_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w3.T), "x3")

        # AiM AF EWMUL
        x1_sigmoid = torch.sigmoid(x1_aim)
        if self.pim_compute:
            # compare(x1_sigmoid_aim, torch.sigmoid(x1_aim), "x1 sigmoid")
            iteration_required = x1_aim.shape[-1] > self.channels_per_block * (self.num_banks // 4) * self.DRAM_column
            if iteration_required:
                iteration_0 = self.total_banks // 4 * 1024
                self.dic_shape["x1_bank_group_0"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.dic_shape["x1_bank_group_1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_1", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], False)
                x1_silu_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], False)
                x1_silu = torch.cat((x1_silu_0, x1_silu_1), dim=2)
            else:
                self.dic_shape["x1_bank_group"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], False)
            compare(x1_silu[0][0], (x1_aim * x1_sigmoid)[0][0], "x1_silu")
        else:
            compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
            self.dic_shape["x1_sigmoid"] = self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)
            x1_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_sigmoid_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu = self.Vector_Vector_EWMUL(x1_aim_load, x1_sigmoid_load)
            self.store_to_DRAM_multi_channel(x1_silu[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)

        # AiM EWMUL
        if self.pim_compute:
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    if iteration_required:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                    else:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
            if iteration_required:
                self.store_to_DRAM_multi_channel(x3_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x3_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], self.trace_activation)
                ffn_vector_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], self.trace_activation)
                ffn_vector = torch.cat((ffn_vector_0, ffn_vector_1), dim=2)
            else:
                self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], self.trace_activation)
        else:
            x3_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x3_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            ffn_vector = self.Vector_Vector_EWMUL(x1_silu_load, x3_aim_load)
        compare(ffn_vector[0][0], (F.silu(x1_aim) * x3_aim)[0][0], "ffn_vector")

        # AiM MAC BK x GB
        if self.pim_compute:
            ffn_aim = self.Vector_Matrix_Mul_weight_pim(ffn_vector[0][0], self.w2_row_index, self.w1.shape[0], self.w2.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight").reshape(bsz, 1, -1)
        else:
            w2_aim = self.load_from_DRAM_multi_channel(self.w2.shape, self.w2_row_index, self.mode["weights"], self.dic_shape["w2"][0], False)
            ffn_aim = self.Vector_Matrix_Mul_multithreads(ffn_vector[0][0], w2_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["ffn_bank_group"] = self.store_to_DRAM_multi_channel(ffn_aim[0][0], self.ffn_row_index, "vector_bank_group_1", False)
        compare(ffn_aim[0][0], self.ffn[0][0], "Vector_Matrix_Mul ffn")

        # AiM EWADD
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.ffn_row_index, "vector_bank_group_0", False)
        sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.ffn_row_index, "vector_bank_group_0", self.dic_shape["ffn_bank_group"][0], False)
        ffn_load = self.load_from_DRAM_multi_channel(self.ffn.shape, self.ffn_row_index, "vector_bank_group_1", self.dic_shape["ffn_bank_group"][0], False)
        out_aim = self.Vector_Vector_EWADD(sa_load, ffn_load)
        self.dic_shape["out_bank_group"] = self.store_to_DRAM_multi_channel(out_aim[0][0], self.ffn_row_index, "vector_bank_group_2", False)

        return out_aim
    
    def trace_only(self):
        bsz, _, _ = self.x.shape
        seqlen = self.seqlen
        total_banks = self.total_banks
        if self.model_parallel:
            FC_total_banks = total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = total_banks
            channels_required = self.channels_per_block
        channel_multi_transformer_block_required = self.num_channels // channels_required * channels_required
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)

        input_vector_neighbor_bank_length = (self.dim - 1) // (self.total_banks // 2) + 1
        input_vector_neighbor_bank_utilized_banks = (self.dim - 1) // input_vector_neighbor_bank_length + 1
        if self.trace_norm:
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 0, self.x_row_index, input_vector_neighbor_bank_length)
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 1, self.x_row_index, input_vector_neighbor_bank_length)

        # RMSNorm   x.pow   MAC_ABK
        input_vector_MAB_BK_BK_length = (self.dim - 1) // (total_banks // 2) + 1
        if self.trace_norm:
            self.WR_BIAS_only_trace(channel_lst)
            self.MAC_ABK_only_trace(channel_lst, self.x_row_index, (input_vector_MAB_BK_BK_length - 1) // self.burst_length + 1, "breakdown_sa_pow")
            self.RD_MAC_only_trace(channel_lst)

        # CXL Port  
        # Reduction of dim // 16 intermidiate sum read from MAC
        # Broadcast a scalar to vector and store it for EWMUL
        input_vector_EWMUL_length = (self.dim - 1) // (total_banks // 4) + 1
        input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1
        if self.trace_norm:
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.x_copy_row_index, input_vector_EWMUL_length)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.x_copy_row_index, input_vector_EWMUL_length)

            # RMSNorm   EWMUL
            self.EWMUL_only_trace(channel_lst, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            for bank in range(self.num_banks):
                bank_group_index = 2
                if bank % 4 == bank_group_index:
                    self.COPY_BK_GB_only_trace(channel_lst, bank, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
                    self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
            self.EWMUL_only_trace(channel_lst, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            # Read RMSNorm result vector to GPR
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim // self.burst_length
            self.load_from_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.SANorm_row_index, input_vector_EWMUL_length)
            self.SYNC_only_trace()

        # K/Q/V GEMV
        if self.trace_fc_kqvo:
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wq_row_index, self.dim, self.head_dim * self.n_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wk_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wv_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")

            # CXL Port
            # Store re-mapped xq/xk for EWMUL
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            # Rotary embedding
            self.EWMUL_only_trace(channel_lst, self.xq_row_index, self.dim // self.burst_length)
            self.EWMUL_only_trace(channel_lst, self.xk_row_index, self.dim // self.n_repeat // self.burst_length)
            # Load rotary embedding results
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)

        if self.trace_attention:
            # Store xk
            seq = seqlen - 1
            dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
            rows = self.head_dim * self.n_kv_heads // self.DRAM_column
            for row in range(rows):
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.DRAM_column // self.burst_length
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + tb * channels_required, bank_index, self.cache_k_row_index + seq // self.FC_total_banks * rows + row, self.DRAM_column)
            # Store xv
            if self.intra_device_attention:
                num_rows_per_seq = (seq - 1) // self.DRAM_column + 1
                row_offset = num_rows_per_seq - 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                dim_iteration = self.head_dim // self.num_banks
                for head_index_per_bank in range(num_heads_per_bank):
                    row_current_head = self.cache_v_row_index + (rows_per_dim * dim_iteration) * head_index_per_bank
                    for dim_iter in range(dim_iteration):
                        for channel in range(channels_required):
                            head = channel * num_heads_per_bank + head_index_per_bank
                            if head > self.n_kv_heads - 1:
                                break
                            # for bank in range(self.num_banks):
                            #     dim = dim_iter * self.num_banks + bank
                            #     self.W_MEM_only_trace(channel, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 1)
                            self.WR_ABK_only_trace(channel, row_current_head + dim_iter * rows_per_dim + row_offset, 1)
            else:
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                for channel in range(channels_required):
                    if banks_per_head < self.num_banks:
                        # print("banks_per_head", banks_per_head)
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            row_offset = rows_per_seq - 1
                            if self.trace_attention and channel < self.num_channels:
                                self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, 1)
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel % channels_per_head) * self.num_banks + bank
                                self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            if (channel % channels_per_head) // channels_per_row_offset == rows_per_seq - 1:
                                row_offset = rows_per_seq - 1
                                if self.trace_attention and channel < self.num_channels:
                                    self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset // rows_per_seq_iteration, 1)
                                for bank in range(self.num_banks):
                                    dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                                    self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset // rows_per_seq_iteration, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)

        else:
            cache_v_size = torch.Size([bsz, self.n_kv_heads, self.head_dim, -1])
            cache_k = self.load_from_DRAM_multi_channel(self.cache_k.shape, self.cache_k_row_index, self.mode["cache_k"], self.cache_k.shape[1], False)
            cache_v = self.load_from_DRAM_multi_channel(cache_v_size, self.cache_v_row_index, self.mode["cache_v"], self.cache_k.shape[1], False).transpose(2, 3).transpose(1, 2).reshape(bsz, -1, self.n_kv_heads, self.head_dim)
            compare(cache_k, self.cache_k, "cache v old")
            compare(cache_v, self.cache_v, "cache k old")
            cache_k[:bsz, self.start_pos : self.start_pos + 1] = xk_aim
            cache_v[:bsz, self.start_pos : self.start_pos + 1] = xv_aim

            keys_aim = cache_k[:bsz, : self.start_pos + 1]
            values_aim = cache_v[:bsz, : self.start_pos + 1]
            if self.GQA:
                keys_aim = repeat_kv(keys_aim, self.n_repeat)
                values_aim = repeat_kv(values_aim, self.n_repeat)
            xq_aim_load = xq_aim_load.transpose(1, 2)
            keys_aim = keys_aim.transpose(1, 2).transpose(2, 3)
            values_aim = values_aim.transpose(1, 2)

        # AiM MAC BK x GB
        if self.pim_compute:
            scores_aim = self.Vector_Matrix_Mul_score_pim(self.xq_row_index, self.cache_k_row_index, self.trace_attention, "breakdown_sa_score")

            if debug:
                self.cache_k[:bsz, self.start_pos : self.start_pos + seqlen] = xk_aim
                self.cache_v[:bsz, self.start_pos : self.start_pos + seqlen] = xv_aim
                keys = self.cache_k[:bsz, : self.start_pos + seqlen]
                values = self.cache_v[:bsz, : self.start_pos + seqlen]
                if self.GQA:
                    keys = repeat_kv(keys, self.n_repeat)
                    values = repeat_kv(values, self.n_repeat)
                xq = xq_aim.transpose(1, 2)
                keys = keys.transpose(1, 2).transpose(2, 3)
                compare(scores_aim, torch.matmul(xq, keys), "Vector_Matrix_Mul score")
        else:
            scores_aim = []
            for i in range(self.n_heads):
                scores_aim.append(self.Vector_Matrix_Mul(xq_aim_load[0][i][0], keys_aim[0][i], False))
            scores_aim = torch.tensor(scores_aim).reshape(bsz, self.n_heads, 1, -1)

        # CXL Ports
        head_dim_reciprocal = torch.full(scores_aim.shape, 1 / math.sqrt(self.head_dim))
        self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(head_dim_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in channel_lst:
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
        else:
            scores_aim_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            head_dim_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_aim_load, head_dim_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)

        # CXL Ports
        scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        if self.pim_compute and debug:
            scores = torch.matmul(xq, keys) / math.sqrt(self.head_dim)
            compare(scores_aim, scores, "Vector_Matrix_Mul score / head_dim")

        scores_exp = torch.exp(scores_aim)
        scores_exp_sum_reciprocal = 1 / torch.sum(scores_exp, dim=-1, keepdim=True)
        scores_exp_sum_reciprocal = torch.cat([scores_exp_sum_reciprocal for i in range(scores_exp.shape[-1])], dim=-1)
        self.store_to_DRAM_multi_channel(scores_exp.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(scores_exp_sum_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in range(channels_required):
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
            scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        else:
            scores_exp_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            scores_exp_sum_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_exp_load, scores_exp_sum_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)
        compare(scores_aim, self.scores, "SoftMax scores")

        # AiM MAC BK x GB
        if self.pim_compute:
            output_aim = self.Vector_Matrix_Mul_output_pim(scores_aim, self.cache_v_row_index, self.trace_attention, "breakdown_sa_output").reshape(bsz, 1, -1)
        else:
            output_aim = []
            for i in range(self.n_heads):
                output_aim.append(self.Vector_Matrix_Mul(scores_aim[0][i][0], values_aim[0][i], False))
            output_aim = torch.tensor(output_aim).reshape(bsz, 1, -1)
        compare(output_aim[0][0], self.output[0][0], "Vector_Matrix_Mul output")

        # CXL Ports
        self.dic_shape["output"] = self.store_to_DRAM_multi_channel(output_aim[0][0], self.output_row_index, self.mode["vector"], False)

        # AiM MAC BK x GB
        if self.pim_compute:
            sa_aim = self.Vector_Matrix_Mul_weight_pim(output_aim[0][0], self.wo_row_index, self.dim, self.wo.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
        else:
            wo_aim = self.load_from_DRAM_multi_channel(self.wo.shape, self.wo_row_index, self.mode["weights"], self.dic_shape["wo"][0], False)
            sa_aim = self.Vector_Matrix_Mul_multithreads(output_aim[0][0], wo_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_0", False)
        compare(sa_aim[0][0], self.sa[0][0], "Vector_Matrix_Mul sa")

        # CXL Ports
        sa_aim_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
        x_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
        sa_aim = self.Vector_Vector_EWADD(x_load, sa_aim_load)
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_2", False)

        return sa_aim
    
    def FFN(self, sa):
        compare(sa[0][0], self.h[0][0], "h")
        RMSNorm_sa = RMSNorm(sa, self.FFNNorm)
        x1 = F.linear(RMSNorm_sa, self.w1)
        x3 = F.linear(RMSNorm_sa, self.w3)
        ffn = F.linear(F.silu(x1) * x3, self.w2)
        compare(ffn[0][0], self.ffn[0][0], "ffn")
        out = sa + ffn
        return out
    
    def FFN_aim(self, sa_aim):
        bsz, _, _ = self.sa.shape
        if self.model_parallel:
            FC_total_banks = self.total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = self.total_banks
            channels_required = self.channels_per_block
        channel_lst = [channel for channel in range(channels_required)]

        # AiM MAC BK x BK
        if self.pim_compute:
            self.dic_shape["sa_neighbor_bank"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_0", self.trace_norm)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_1", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            sa_pow_sum = 0
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.WR_BIAS(0, channel, channels_required, 0, [0 for bank in range(self.num_banks)], op_trace)
                op_size = (self.dic_shape["sa_neighbor_bank"][0] - 1) // self.burst_length + 1
                self.MAC_BK_BK(0, channel, channels_required, self.sa_copy_row_index, 0, 0, op_size, op_trace)
                sa_pow_sum += sum(self.RD_MAC(0, channel, channels_required, 0, op_trace))    # CXL ports
        else:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", False)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_1", False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            sa_copy_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_pow_sum = self.Vector_Vector_Mul(sa_load[0][0], sa_copy_load[0][0], False)

        # CXL Ports
        compare(sa_pow_sum, sa_aim.pow(2).sum(), "sa pow")
        norm = torch.rsqrt(sa_pow_sum / self.dim + 1e-5)
        norm_tensor = torch.full(sa_aim.shape, norm)
        self.store_to_DRAM_multi_channel(norm_tensor[0][0], self.sa_copy_row_index, "vector_bank_group_1", self.trace_norm)
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length

        # AiM EWMUL
        if self.pim_compute:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.sa_copy_row_index, 0, op_size, op_trace)
        else:
            norm_tensor_load = self.load_from_DRAM_multi_channel(self.x.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            norm_sa = self.Vector_Vector_EWMUL(sa_load, norm_tensor_load)
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        # AiM EWMUL
        if self.pim_compute:
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_norm
                    self.COPY_BK_GB(0, channel, channels_required, bank, self.sa_copy_row_index, 0, op_size, op_trace)
                    self.COPY_GB_BK(0, channel, channels_required, bank-1, self.FFNNorm_row_index, 0, op_size, op_trace)
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.FFNNorm_row_index, 0, op_size, op_trace)
            FFNNorm_sa_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_2", self.dic_shape["FFNNorm"][0], self.trace_norm)
        else:
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_1", False)
            norm_sa_load = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            FFNNorm_load = self.load_from_DRAM_multi_channel(self.FFNNorm.shape, self.FFNNorm_row_index, "vector_bank_group_0", self.dic_shape["FFNNorm"][0], False)
            FFNNorm_sa_aim = self.Vector_Vector_EWMUL(norm_sa_load, FFNNorm_load)
            self.store_to_DRAM_multi_channel(FFNNorm_sa_aim[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        bsz, _, _ = FFNNorm_sa_aim.shape
        compare(FFNNorm_sa_aim[0][0], RMSNorm(sa_aim[0][0], self.FFNNorm), "FFNNorm_sa_aim")

        # AiM MAC BK x GB
        if self.pim_compute:
            x1_aim, x1_sigmoid_aim = self.Vector_Matrix_Mul_weight_af_pim(FFNNorm_sa_aim[0][0], self.w1_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")
            x1_aim = x1_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x1_sigmoid_aim = x1_sigmoid_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_weight_pim(FFNNorm_sa_aim[0][0], self.w3_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")[:self.w3.shape[0]].reshape(bsz, 1, -1)
        else:
            w1_aim = self.load_from_DRAM_multi_channel(self.w1.shape, self.w1_row_index, self.mode["weights"], self.dic_shape["w1"][0], False)
            w3_aim = self.load_from_DRAM_multi_channel(self.w3.shape, self.w3_row_index, self.mode["weights"], self.dic_shape["w3"][0], False)
            x1_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w1_aim.T).reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w3_aim.T).reshape(bsz, 1, -1)
            self.dic_shape["x1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_row_index, self.mode["vector"], False)
            self.dic_shape["x3"] = self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x3_row_index, self.mode["vector"], False)
        compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
        compare(x3_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w3.T), "x3")

        # AiM AF EWMUL
        x1_sigmoid = torch.sigmoid(x1_aim)
        if self.pim_compute:
            # compare(x1_sigmoid_aim, torch.sigmoid(x1_aim), "x1 sigmoid")
            iteration_required = x1_aim.shape[-1] > self.channels_per_block * (self.num_banks // 4) * self.DRAM_column
            if iteration_required:
                iteration_0 = self.total_banks // 4 * 1024
                self.dic_shape["x1_bank_group_0"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.dic_shape["x1_bank_group_1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_1", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], False)
                x1_silu_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], False)
                x1_silu = torch.cat((x1_silu_0, x1_silu_1), dim=2)
            else:
                self.dic_shape["x1_bank_group"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], False)
            compare(x1_silu[0][0], (x1_aim * x1_sigmoid)[0][0], "x1_silu")
        else:
            compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
            self.dic_shape["x1_sigmoid"] = self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)
            x1_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_sigmoid_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu = self.Vector_Vector_EWMUL(x1_aim_load, x1_sigmoid_load)
            self.store_to_DRAM_multi_channel(x1_silu[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)

        # AiM EWMUL
        if self.pim_compute:
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    if iteration_required:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                    else:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
            if iteration_required:
                self.store_to_DRAM_multi_channel(x3_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x3_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], self.trace_activation)
                ffn_vector_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], self.trace_activation)
                ffn_vector = torch.cat((ffn_vector_0, ffn_vector_1), dim=2)
            else:
                self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], self.trace_activation)
        else:
            x3_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x3_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            ffn_vector = self.Vector_Vector_EWMUL(x1_silu_load, x3_aim_load)
        compare(ffn_vector[0][0], (F.silu(x1_aim) * x3_aim)[0][0], "ffn_vector")

        # AiM MAC BK x GB
        if self.pim_compute:
            ffn_aim = self.Vector_Matrix_Mul_weight_pim(ffn_vector[0][0], self.w2_row_index, self.w1.shape[0], self.w2.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight").reshape(bsz, 1, -1)
        else:
            w2_aim = self.load_from_DRAM_multi_channel(self.w2.shape, self.w2_row_index, self.mode["weights"], self.dic_shape["w2"][0], False)
            ffn_aim = self.Vector_Matrix_Mul_multithreads(ffn_vector[0][0], w2_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["ffn_bank_group"] = self.store_to_DRAM_multi_channel(ffn_aim[0][0], self.ffn_row_index, "vector_bank_group_1", False)
        compare(ffn_aim[0][0], self.ffn[0][0], "Vector_Matrix_Mul ffn")

        # AiM EWADD
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.ffn_row_index, "vector_bank_group_0", False)
        sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.ffn_row_index, "vector_bank_group_0", self.dic_shape["ffn_bank_group"][0], False)
        ffn_load = self.load_from_DRAM_multi_channel(self.ffn.shape, self.ffn_row_index, "vector_bank_group_1", self.dic_shape["ffn_bank_group"][0], False)
        out_aim = self.Vector_Vector_EWADD(sa_load, ffn_load)
        self.dic_shape["out_bank_group"] = self.store_to_DRAM_multi_channel(out_aim[0][0], self.ffn_row_index, "vector_bank_group_2", False)

        return out_aim
    
    def trace_only(self):
        bsz, _, _ = self.x.shape
        seqlen = self.seqlen
        total_banks = self.total_banks
        if self.model_parallel:
            FC_total_banks = total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = total_banks
            channels_required = self.channels_per_block
        channel_multi_transformer_block_required = self.num_channels // channels_required * channels_required
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)

        input_vector_neighbor_bank_length = (self.dim - 1) // (self.total_banks // 2) + 1
        input_vector_neighbor_bank_utilized_banks = (self.dim - 1) // input_vector_neighbor_bank_length + 1
        if self.trace_norm:
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 0, self.x_row_index, input_vector_neighbor_bank_length)
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 1, self.x_row_index, input_vector_neighbor_bank_length)

        # RMSNorm   x.pow   MAC_ABK
        input_vector_MAB_BK_BK_length = (self.dim - 1) // (total_banks // 2) + 1
        if self.trace_norm:
            self.WR_BIAS_only_trace(channel_lst)
            self.MAC_ABK_only_trace(channel_lst, self.x_row_index, (input_vector_MAB_BK_BK_length - 1) // self.burst_length + 1, "breakdown_sa_pow")
            self.RD_MAC_only_trace(channel_lst)

        # CXL Port  
        # Reduction of dim // 16 intermidiate sum read from MAC
        # Broadcast a scalar to vector and store it for EWMUL
        input_vector_EWMUL_length = (self.dim - 1) // (total_banks // 4) + 1
        input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1
        if self.trace_norm:
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.x_copy_row_index, input_vector_EWMUL_length)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.x_copy_row_index, input_vector_EWMUL_length)

            # RMSNorm   EWMUL
            self.EWMUL_only_trace(channel_lst, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            for bank in range(self.num_banks):
                bank_group_index = 2
                if bank % 4 == bank_group_index:
                    self.COPY_BK_GB_only_trace(channel_lst, bank, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
                    self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
            self.EWMUL_only_trace(channel_lst, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            # Read RMSNorm result vector to GPR
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim // self.burst_length
            self.load_from_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.SANorm_row_index, input_vector_EWMUL_length)
            self.SYNC_only_trace()

        # K/Q/V GEMV
        if self.trace_fc_kqvo:
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wq_row_index, self.dim, self.head_dim * self.n_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wk_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wv_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")

            # CXL Port
            # Store re-mapped xq/xk for EWMUL
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            # Rotary embedding
            self.EWMUL_only_trace(channel_lst, self.xq_row_index, self.dim // self.burst_length)
            self.EWMUL_only_trace(channel_lst, self.xk_row_index, self.dim // self.n_repeat // self.burst_length)
            # Load rotary embedding results
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)

        if self.trace_attention:
            # Store xk
            seq = seqlen - 1
            dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
            rows = self.head_dim * self.n_kv_heads // self.DRAM_column
            for row in range(rows):
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.DRAM_column // self.burst_length
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + tb * channels_required, bank_index, self.cache_k_row_index + seq // self.FC_total_banks * rows + row, self.DRAM_column)
            # Store xv
            if self.intra_device_attention:
                num_rows_per_seq = (seq - 1) // self.DRAM_column + 1
                row_offset = num_rows_per_seq - 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                dim_iteration = self.head_dim // self.num_banks
                for head_index_per_bank in range(num_heads_per_bank):
                    row_current_head = self.cache_v_row_index + (rows_per_dim * dim_iteration) * head_index_per_bank
                    for dim_iter in range(dim_iteration):
                        for channel in range(channels_required):
                            head = channel * num_heads_per_bank + head_index_per_bank
                            if head > self.n_kv_heads - 1:
                                break
                            # for bank in range(self.num_banks):
                            #     dim = dim_iter * self.num_banks + bank
                            #     self.W_MEM_only_trace(channel, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 1)
                            self.WR_ABK_only_trace(channel, row_current_head + dim_iter * rows_per_dim + row_offset, 1)
            else:
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                for channel in range(channels_required):
                    if banks_per_head < self.num_banks:
                        # print("banks_per_head", banks_per_head)
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            row_offset = rows_per_seq - 1
                            if self.trace_attention and channel < self.num_channels:
                                self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, 1)
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel % channels_per_head) * self.num_banks + bank
                                self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            if (channel % channels_per_head) // channels_per_row_offset == rows_per_seq - 1:
                                row_offset = rows_per_seq - 1
                                if self.trace_attention and channel < self.num_channels:
                                    self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset // rows_per_seq_iteration, 1)
                                for bank in range(self.num_banks):
                                    dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                                    self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset // rows_per_seq_iteration, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)

        else:
            cache_v_size = torch.Size([bsz, self.n_kv_heads, self.head_dim, -1])
            cache_k = self.load_from_DRAM_multi_channel(self.cache_k.shape, self.cache_k_row_index, self.mode["cache_k"], self.cache_k.shape[1], False)
            cache_v = self.load_from_DRAM_multi_channel(cache_v_size, self.cache_v_row_index, self.mode["cache_v"], self.cache_k.shape[1], False).transpose(2, 3).transpose(1, 2).reshape(bsz, -1, self.n_kv_heads, self.head_dim)
            compare(cache_k, self.cache_k, "cache v old")
            compare(cache_v, self.cache_v, "cache k old")
            cache_k[:bsz, self.start_pos : self.start_pos + 1] = xk_aim
            cache_v[:bsz, self.start_pos : self.start_pos + 1] = xv_aim

            keys_aim = cache_k[:bsz, : self.start_pos + 1]
            values_aim = cache_v[:bsz, : self.start_pos + 1]
            if self.GQA:
                keys_aim = repeat_kv(keys_aim, self.n_repeat)
                values_aim = repeat_kv(values_aim, self.n_repeat)
            xq_aim_load = xq_aim_load.transpose(1, 2)
            keys_aim = keys_aim.transpose(1, 2).transpose(2, 3)
            values_aim = values_aim.transpose(1, 2)

        # AiM MAC BK x GB
        if self.pim_compute:
            scores_aim = self.Vector_Matrix_Mul_score_pim(self.xq_row_index, self.cache_k_row_index, self.trace_attention, "breakdown_sa_score")

            if debug:
                self.cache_k[:bsz, self.start_pos : self.start_pos + seqlen] = xk_aim
                self.cache_v[:bsz, self.start_pos : self.start_pos + seqlen] = xv_aim
                keys = self.cache_k[:bsz, : self.start_pos + seqlen]
                values = self.cache_v[:bsz, : self.start_pos + seqlen]
                if self.GQA:
                    keys = repeat_kv(keys, self.n_repeat)
                    values = repeat_kv(values, self.n_repeat)
                xq = xq_aim.transpose(1, 2)
                keys = keys.transpose(1, 2).transpose(2, 3)
                compare(scores_aim, torch.matmul(xq, keys), "Vector_Matrix_Mul score")
        else:
            scores_aim = []
            for i in range(self.n_heads):
                scores_aim.append(self.Vector_Matrix_Mul(xq_aim_load[0][i][0], keys_aim[0][i], False))
            scores_aim = torch.tensor(scores_aim).reshape(bsz, self.n_heads, 1, -1)

        # CXL Ports
        head_dim_reciprocal = torch.full(scores_aim.shape, 1 / math.sqrt(self.head_dim))
        self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(head_dim_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in channel_lst:
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
        else:
            scores_aim_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            head_dim_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_aim_load, head_dim_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)

        # CXL Ports
        scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        if self.pim_compute and debug:
            scores = torch.matmul(xq, keys) / math.sqrt(self.head_dim)
            compare(scores_aim, scores, "Vector_Matrix_Mul score / head_dim")

        scores_exp = torch.exp(scores_aim)
        scores_exp_sum_reciprocal = 1 / torch.sum(scores_exp, dim=-1, keepdim=True)
        scores_exp_sum_reciprocal = torch.cat([scores_exp_sum_reciprocal for i in range(scores_exp.shape[-1])], dim=-1)
        self.store_to_DRAM_multi_channel(scores_exp.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(scores_exp_sum_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in range(channels_required):
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
            scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        else:
            scores_exp_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            scores_exp_sum_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_exp_load, scores_exp_sum_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)
        compare(scores_aim, self.scores, "SoftMax scores")

        # AiM MAC BK x GB
        if self.pim_compute:
            output_aim = self.Vector_Matrix_Mul_output_pim(scores_aim, self.cache_v_row_index, self.trace_attention, "breakdown_sa_output").reshape(bsz, 1, -1)
        else:
            output_aim = []
            for i in range(self.n_heads):
                output_aim.append(self.Vector_Matrix_Mul(scores_aim[0][i][0], values_aim[0][i], False))
            output_aim = torch.tensor(output_aim).reshape(bsz, 1, -1)
        compare(output_aim[0][0], self.output[0][0], "Vector_Matrix_Mul output")

        # CXL Ports
        self.dic_shape["output"] = self.store_to_DRAM_multi_channel(output_aim[0][0], self.output_row_index, self.mode["vector"], False)

        # AiM MAC BK x GB
        if self.pim_compute:
            sa_aim = self.Vector_Matrix_Mul_weight_pim(output_aim[0][0], self.wo_row_index, self.dim, self.wo.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
        else:
            wo_aim = self.load_from_DRAM_multi_channel(self.wo.shape, self.wo_row_index, self.mode["weights"], self.dic_shape["wo"][0], False)
            sa_aim = self.Vector_Matrix_Mul_multithreads(output_aim[0][0], wo_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_0", False)
        compare(sa_aim[0][0], self.sa[0][0], "Vector_Matrix_Mul sa")

        # CXL Ports
        sa_aim_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
        x_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
        sa_aim = self.Vector_Vector_EWADD(x_load, sa_aim_load)
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_2", False)

        return sa_aim
    
    def FFN(self, sa):
        compare(sa[0][0], self.h[0][0], "h")
        RMSNorm_sa = RMSNorm(sa, self.FFNNorm)
        x1 = F.linear(RMSNorm_sa, self.w1)
        x3 = F.linear(RMSNorm_sa, self.w3)
        ffn = F.linear(F.silu(x1) * x3, self.w2)
        compare(ffn[0][0], self.ffn[0][0], "ffn")
        out = sa + ffn
        return out
    
    def FFN_aim(self, sa_aim):
        bsz, _, _ = self.sa.shape
        if self.model_parallel:
            FC_total_banks = self.total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = self.total_banks
            channels_required = self.channels_per_block
        channel_lst = [channel for channel in range(channels_required)]

        # AiM MAC BK x BK
        if self.pim_compute:
            self.dic_shape["sa_neighbor_bank"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_0", self.trace_norm)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_1", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            sa_pow_sum = 0
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.WR_BIAS(0, channel, channels_required, 0, [0 for bank in range(self.num_banks)], op_trace)
                op_size = (self.dic_shape["sa_neighbor_bank"][0] - 1) // self.burst_length + 1
                self.MAC_BK_BK(0, channel, channels_required, self.sa_copy_row_index, 0, 0, op_size, op_trace)
                sa_pow_sum += sum(self.RD_MAC(0, channel, channels_required, 0, op_trace))    # CXL ports
        else:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", False)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_1", False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            sa_copy_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_pow_sum = self.Vector_Vector_Mul(sa_load[0][0], sa_copy_load[0][0], False)

        # CXL Ports
        compare(sa_pow_sum, sa_aim.pow(2).sum(), "sa pow")
        norm = torch.rsqrt(sa_pow_sum / self.dim + 1e-5)
        norm_tensor = torch.full(sa_aim.shape, norm)
        self.store_to_DRAM_multi_channel(norm_tensor[0][0], self.sa_copy_row_index, "vector_bank_group_1", self.trace_norm)
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length

        # AiM EWMUL
        if self.pim_compute:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.sa_copy_row_index, 0, op_size, op_trace)
        else:
            norm_tensor_load = self.load_from_DRAM_multi_channel(self.x.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            norm_sa = self.Vector_Vector_EWMUL(sa_load, norm_tensor_load)
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        # AiM EWMUL
        if self.pim_compute:
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_norm
                    self.COPY_BK_GB(0, channel, channels_required, bank, self.sa_copy_row_index, 0, op_size, op_trace)
                    self.COPY_GB_BK(0, channel, channels_required, bank-1, self.FFNNorm_row_index, 0, op_size, op_trace)
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.FFNNorm_row_index, 0, op_size, op_trace)
            FFNNorm_sa_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_2", self.dic_shape["FFNNorm"][0], self.trace_norm)
        else:
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_1", False)
            norm_sa_load = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            FFNNorm_load = self.load_from_DRAM_multi_channel(self.FFNNorm.shape, self.FFNNorm_row_index, "vector_bank_group_0", self.dic_shape["FFNNorm"][0], False)
            FFNNorm_sa_aim = self.Vector_Vector_EWMUL(norm_sa_load, FFNNorm_load)
            self.store_to_DRAM_multi_channel(FFNNorm_sa_aim[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        bsz, _, _ = FFNNorm_sa_aim.shape
        compare(FFNNorm_sa_aim[0][0], RMSNorm(sa_aim[0][0], self.FFNNorm), "FFNNorm_sa_aim")

        # AiM MAC BK x GB
        if self.pim_compute:
            x1_aim, x1_sigmoid_aim = self.Vector_Matrix_Mul_weight_af_pim(FFNNorm_sa_aim[0][0], self.w1_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")
            x1_aim = x1_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x1_sigmoid_aim = x1_sigmoid_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_weight_pim(FFNNorm_sa_aim[0][0], self.w3_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")[:self.w3.shape[0]].reshape(bsz, 1, -1)
        else:
            w1_aim = self.load_from_DRAM_multi_channel(self.w1.shape, self.w1_row_index, self.mode["weights"], self.dic_shape["w1"][0], False)
            w3_aim = self.load_from_DRAM_multi_channel(self.w3.shape, self.w3_row_index, self.mode["weights"], self.dic_shape["w3"][0], False)
            x1_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w1_aim.T).reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w3_aim.T).reshape(bsz, 1, -1)
            self.dic_shape["x1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_row_index, self.mode["vector"], False)
            self.dic_shape["x3"] = self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x3_row_index, self.mode["vector"], False)
        compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
        compare(x3_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w3.T), "x3")

        # AiM AF EWMUL
        x1_sigmoid = torch.sigmoid(x1_aim)
        if self.pim_compute:
            # compare(x1_sigmoid_aim, torch.sigmoid(x1_aim), "x1 sigmoid")
            iteration_required = x1_aim.shape[-1] > self.channels_per_block * (self.num_banks // 4) * self.DRAM_column
            if iteration_required:
                iteration_0 = self.total_banks // 4 * 1024
                self.dic_shape["x1_bank_group_0"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.dic_shape["x1_bank_group_1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_1", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], False)
                x1_silu_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], False)
                x1_silu = torch.cat((x1_silu_0, x1_silu_1), dim=2)
            else:
                self.dic_shape["x1_bank_group"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], False)
            compare(x1_silu[0][0], (x1_aim * x1_sigmoid)[0][0], "x1_silu")
        else:
            compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
            self.dic_shape["x1_sigmoid"] = self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)
            x1_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_sigmoid_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu = self.Vector_Vector_EWMUL(x1_aim_load, x1_sigmoid_load)
            self.store_to_DRAM_multi_channel(x1_silu[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)

        # AiM EWMUL
        if self.pim_compute:
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    if iteration_required:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                    else:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
            if iteration_required:
                self.store_to_DRAM_multi_channel(x3_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x3_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], self.trace_activation)
                ffn_vector_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], self.trace_activation)
                ffn_vector = torch.cat((ffn_vector_0, ffn_vector_1), dim=2)
            else:
                self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], self.trace_activation)
        else:
            x3_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x3_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            ffn_vector = self.Vector_Vector_EWMUL(x1_silu_load, x3_aim_load)
        compare(ffn_vector[0][0], (F.silu(x1_aim) * x3_aim)[0][0], "ffn_vector")

        # AiM MAC BK x GB
        if self.pim_compute:
            ffn_aim = self.Vector_Matrix_Mul_weight_pim(ffn_vector[0][0], self.w2_row_index, self.w1.shape[0], self.w2.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight").reshape(bsz, 1, -1)
        else:
            w2_aim = self.load_from_DRAM_multi_channel(self.w2.shape, self.w2_row_index, self.mode["weights"], self.dic_shape["w2"][0], False)
            ffn_aim = self.Vector_Matrix_Mul_multithreads(ffn_vector[0][0], w2_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["ffn_bank_group"] = self.store_to_DRAM_multi_channel(ffn_aim[0][0], self.ffn_row_index, "vector_bank_group_1", False)
        compare(ffn_aim[0][0], self.ffn[0][0], "Vector_Matrix_Mul ffn")

        # AiM EWADD
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.ffn_row_index, "vector_bank_group_0", False)
        sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.ffn_row_index, "vector_bank_group_0", self.dic_shape["ffn_bank_group"][0], False)
        ffn_load = self.load_from_DRAM_multi_channel(self.ffn.shape, self.ffn_row_index, "vector_bank_group_1", self.dic_shape["ffn_bank_group"][0], False)
        out_aim = self.Vector_Vector_EWADD(sa_load, ffn_load)
        self.dic_shape["out_bank_group"] = self.store_to_DRAM_multi_channel(out_aim[0][0], self.ffn_row_index, "vector_bank_group_2", False)

        return out_aim
    
    def trace_only(self):
        bsz, _, _ = self.x.shape
        seqlen = self.seqlen
        total_banks = self.total_banks
        if self.model_parallel:
            FC_total_banks = total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = total_banks
            channels_required = self.channels_per_block
        channel_multi_transformer_block_required = self.num_channels // channels_required * channels_required
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)

        input_vector_neighbor_bank_length = (self.dim - 1) // (self.total_banks // 2) + 1
        input_vector_neighbor_bank_utilized_banks = (self.dim - 1) // input_vector_neighbor_bank_length + 1
        if self.trace_norm:
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 0, self.x_row_index, input_vector_neighbor_bank_length)
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 1, self.x_row_index, input_vector_neighbor_bank_length)

        # RMSNorm   x.pow   MAC_ABK
        input_vector_MAB_BK_BK_length = (self.dim - 1) // (total_banks // 2) + 1
        if self.trace_norm:
            self.WR_BIAS_only_trace(channel_lst)
            self.MAC_ABK_only_trace(channel_lst, self.x_row_index, (input_vector_MAB_BK_BK_length - 1) // self.burst_length + 1, "breakdown_sa_pow")
            self.RD_MAC_only_trace(channel_lst)

        # CXL Port  
        # Reduction of dim // 16 intermidiate sum read from MAC
        # Broadcast a scalar to vector and store it for EWMUL
        input_vector_EWMUL_length = (self.dim - 1) // (total_banks // 4) + 1
        input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1
        if self.trace_norm:
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.x_copy_row_index, input_vector_EWMUL_length)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.x_copy_row_index, input_vector_EWMUL_length)

            # RMSNorm   EWMUL
            self.EWMUL_only_trace(channel_lst, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            for bank in range(self.num_banks):
                bank_group_index = 2
                if bank % 4 == bank_group_index:
                    self.COPY_BK_GB_only_trace(channel_lst, bank, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
                    self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
            self.EWMUL_only_trace(channel_lst, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            # Read RMSNorm result vector to GPR
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim // self.burst_length
            self.load_from_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.SANorm_row_index, input_vector_EWMUL_length)
            self.SYNC_only_trace()

        # K/Q/V GEMV
        if self.trace_fc_kqvo:
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wq_row_index, self.dim, self.head_dim * self.n_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wk_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wv_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")

            # CXL Port
            # Store re-mapped xq/xk for EWMUL
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            # Rotary embedding
            self.EWMUL_only_trace(channel_lst, self.xq_row_index, self.dim // self.burst_length)
            self.EWMUL_only_trace(channel_lst, self.xk_row_index, self.dim // self.n_repeat // self.burst_length)
            # Load rotary embedding results
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)

        if self.trace_attention:
            # Store xk
            seq = seqlen - 1
            dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
            rows = self.head_dim * self.n_kv_heads // self.DRAM_column
            for row in range(rows):
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.DRAM_column // self.burst_length
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + tb * channels_required, bank_index, self.cache_k_row_index + seq // self.FC_total_banks * rows + row, self.DRAM_column)
            # Store xv
            if self.intra_device_attention:
                num_rows_per_seq = (seq - 1) // self.DRAM_column + 1
                row_offset = num_rows_per_seq - 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                dim_iteration = self.head_dim // self.num_banks
                for head_index_per_bank in range(num_heads_per_bank):
                    row_current_head = self.cache_v_row_index + (rows_per_dim * dim_iteration) * head_index_per_bank
                    for dim_iter in range(dim_iteration):
                        for channel in range(channels_required):
                            head = channel * num_heads_per_bank + head_index_per_bank
                            if head > self.n_kv_heads - 1:
                                break
                            # for bank in range(self.num_banks):
                            #     dim = dim_iter * self.num_banks + bank
                            #     self.W_MEM_only_trace(channel, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 1)
                            self.WR_ABK_only_trace(channel, row_current_head + dim_iter * rows_per_dim + row_offset, 1)
            else:
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                for channel in range(channels_required):
                    if banks_per_head < self.num_banks:
                        # print("banks_per_head", banks_per_head)
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            row_offset = rows_per_seq - 1
                            if self.trace_attention and channel < self.num_channels:
                                self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, 1)
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel % channels_per_head) * self.num_banks + bank
                                self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            if (channel % channels_per_head) // channels_per_row_offset == rows_per_seq - 1:
                                row_offset = rows_per_seq - 1
                                if self.trace_attention and channel < self.num_channels:
                                    self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset // rows_per_seq_iteration, 1)
                                for bank in range(self.num_banks):
                                    dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                                    self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset // rows_per_seq_iteration, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)

        else:
            cache_v_size = torch.Size([bsz, self.n_kv_heads, self.head_dim, -1])
            cache_k = self.load_from_DRAM_multi_channel(self.cache_k.shape, self.cache_k_row_index, self.mode["cache_k"], self.cache_k.shape[1], False)
            cache_v = self.load_from_DRAM_multi_channel(cache_v_size, self.cache_v_row_index, self.mode["cache_v"], self.cache_k.shape[1], False).transpose(2, 3).transpose(1, 2).reshape(bsz, -1, self.n_kv_heads, self.head_dim)
            compare(cache_k, self.cache_k, "cache v old")
            compare(cache_v, self.cache_v, "cache k old")
            cache_k[:bsz, self.start_pos : self.start_pos + 1] = xk_aim
            cache_v[:bsz, self.start_pos : self.start_pos + 1] = xv_aim

            keys_aim = cache_k[:bsz, : self.start_pos + 1]
            values_aim = cache_v[:bsz, : self.start_pos + 1]
            if self.GQA:
                keys_aim = repeat_kv(keys_aim, self.n_repeat)
                values_aim = repeat_kv(values_aim, self.n_repeat)
            xq_aim_load = xq_aim_load.transpose(1, 2)
            keys_aim = keys_aim.transpose(1, 2).transpose(2, 3)
            values_aim = values_aim.transpose(1, 2)

        # AiM MAC BK x GB
        if self.pim_compute:
            scores_aim = self.Vector_Matrix_Mul_score_pim(self.xq_row_index, self.cache_k_row_index, self.trace_attention, "breakdown_sa_score")

            if debug:
                self.cache_k[:bsz, self.start_pos : self.start_pos + seqlen] = xk_aim
                self.cache_v[:bsz, self.start_pos : self.start_pos + seqlen] = xv_aim
                keys = self.cache_k[:bsz, : self.start_pos + seqlen]
                values = self.cache_v[:bsz, : self.start_pos + seqlen]
                if self.GQA:
                    keys = repeat_kv(keys, self.n_repeat)
                    values = repeat_kv(values, self.n_repeat)
                xq = xq_aim.transpose(1, 2)
                keys = keys.transpose(1, 2).transpose(2, 3)
                compare(scores_aim, torch.matmul(xq, keys), "Vector_Matrix_Mul score")
        else:
            scores_aim = []
            for i in range(self.n_heads):
                scores_aim.append(self.Vector_Matrix_Mul(xq_aim_load[0][i][0], keys_aim[0][i], False))
            scores_aim = torch.tensor(scores_aim).reshape(bsz, self.n_heads, 1, -1)

        # CXL Ports
        head_dim_reciprocal = torch.full(scores_aim.shape, 1 / math.sqrt(self.head_dim))
        self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(head_dim_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in channel_lst:
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
        else:
            scores_aim_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            head_dim_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_aim_load, head_dim_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)

        # CXL Ports
        scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        if self.pim_compute and debug:
            scores = torch.matmul(xq, keys) / math.sqrt(self.head_dim)
            compare(scores_aim, scores, "Vector_Matrix_Mul score / head_dim")

        scores_exp = torch.exp(scores_aim)
        scores_exp_sum_reciprocal = 1 / torch.sum(scores_exp, dim=-1, keepdim=True)
        scores_exp_sum_reciprocal = torch.cat([scores_exp_sum_reciprocal for i in range(scores_exp.shape[-1])], dim=-1)
        self.store_to_DRAM_multi_channel(scores_exp.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(scores_exp_sum_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in range(channels_required):
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
            scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        else:
            scores_exp_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            scores_exp_sum_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_exp_load, scores_exp_sum_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)
        compare(scores_aim, self.scores, "SoftMax scores")

        # AiM MAC BK x GB
        if self.pim_compute:
            output_aim = self.Vector_Matrix_Mul_output_pim(scores_aim, self.cache_v_row_index, self.trace_attention, "breakdown_sa_output").reshape(bsz, 1, -1)
        else:
            output_aim = []
            for i in range(self.n_heads):
                output_aim.append(self.Vector_Matrix_Mul(scores_aim[0][i][0], values_aim[0][i], False))
            output_aim = torch.tensor(output_aim).reshape(bsz, 1, -1)
        compare(output_aim[0][0], self.output[0][0], "Vector_Matrix_Mul output")

        # CXL Ports
        self.dic_shape["output"] = self.store_to_DRAM_multi_channel(output_aim[0][0], self.output_row_index, self.mode["vector"], False)

        # AiM MAC BK x GB
        if self.pim_compute:
            sa_aim = self.Vector_Matrix_Mul_weight_pim(output_aim[0][0], self.wo_row_index, self.dim, self.wo.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
        else:
            wo_aim = self.load_from_DRAM_multi_channel(self.wo.shape, self.wo_row_index, self.mode["weights"], self.dic_shape["wo"][0], False)
            sa_aim = self.Vector_Matrix_Mul_multithreads(output_aim[0][0], wo_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_0", False)
        compare(sa_aim[0][0], self.sa[0][0], "Vector_Matrix_Mul sa")

        # CXL Ports
        sa_aim_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
        x_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
        sa_aim = self.Vector_Vector_EWADD(x_load, sa_aim_load)
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_2", False)

        return sa_aim
    
    def FFN(self, sa):
        compare(sa[0][0], self.h[0][0], "h")
        RMSNorm_sa = RMSNorm(sa, self.FFNNorm)
        x1 = F.linear(RMSNorm_sa, self.w1)
        x3 = F.linear(RMSNorm_sa, self.w3)
        ffn = F.linear(F.silu(x1) * x3, self.w2)
        compare(ffn[0][0], self.ffn[0][0], "ffn")
        out = sa + ffn
        return out
    
    def FFN_aim(self, sa_aim):
        bsz, _, _ = self.sa.shape
        if self.model_parallel:
            FC_total_banks = self.total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = self.total_banks
            channels_required = self.channels_per_block
        channel_lst = [channel for channel in range(channels_required)]

        # AiM MAC BK x BK
        if self.pim_compute:
            self.dic_shape["sa_neighbor_bank"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_0", self.trace_norm)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_1", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            sa_pow_sum = 0
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.WR_BIAS(0, channel, channels_required, 0, [0 for bank in range(self.num_banks)], op_trace)
                op_size = (self.dic_shape["sa_neighbor_bank"][0] - 1) // self.burst_length + 1
                self.MAC_BK_BK(0, channel, channels_required, self.sa_copy_row_index, 0, 0, op_size, op_trace)
                sa_pow_sum += sum(self.RD_MAC(0, channel, channels_required, 0, op_trace))    # CXL ports
        else:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", False)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_1", False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            sa_copy_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_pow_sum = self.Vector_Vector_Mul(sa_load[0][0], sa_copy_load[0][0], False)

        # CXL Ports
        compare(sa_pow_sum, sa_aim.pow(2).sum(), "sa pow")
        norm = torch.rsqrt(sa_pow_sum / self.dim + 1e-5)
        norm_tensor = torch.full(sa_aim.shape, norm)
        self.store_to_DRAM_multi_channel(norm_tensor[0][0], self.sa_copy_row_index, "vector_bank_group_1", self.trace_norm)
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length

        # AiM EWMUL
        if self.pim_compute:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.sa_copy_row_index, 0, op_size, op_trace)
        else:
            norm_tensor_load = self.load_from_DRAM_multi_channel(self.x.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            norm_sa = self.Vector_Vector_EWMUL(sa_load, norm_tensor_load)
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        # AiM EWMUL
        if self.pim_compute:
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_norm
                    self.COPY_BK_GB(0, channel, channels_required, bank, self.sa_copy_row_index, 0, op_size, op_trace)
                    self.COPY_GB_BK(0, channel, channels_required, bank-1, self.FFNNorm_row_index, 0, op_size, op_trace)
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.FFNNorm_row_index, 0, op_size, op_trace)
            FFNNorm_sa_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_2", self.dic_shape["FFNNorm"][0], self.trace_norm)
        else:
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_1", False)
            norm_sa_load = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            FFNNorm_load = self.load_from_DRAM_multi_channel(self.FFNNorm.shape, self.FFNNorm_row_index, "vector_bank_group_0", self.dic_shape["FFNNorm"][0], False)
            FFNNorm_sa_aim = self.Vector_Vector_EWMUL(norm_sa_load, FFNNorm_load)
            self.store_to_DRAM_multi_channel(FFNNorm_sa_aim[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        bsz, _, _ = FFNNorm_sa_aim.shape
        compare(FFNNorm_sa_aim[0][0], RMSNorm(sa_aim[0][0], self.FFNNorm), "FFNNorm_sa_aim")

        # AiM MAC BK x GB
        if self.pim_compute:
            x1_aim, x1_sigmoid_aim = self.Vector_Matrix_Mul_weight_af_pim(FFNNorm_sa_aim[0][0], self.w1_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")
            x1_aim = x1_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x1_sigmoid_aim = x1_sigmoid_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_weight_pim(FFNNorm_sa_aim[0][0], self.w3_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")[:self.w3.shape[0]].reshape(bsz, 1, -1)
        else:
            w1_aim = self.load_from_DRAM_multi_channel(self.w1.shape, self.w1_row_index, self.mode["weights"], self.dic_shape["w1"][0], False)
            w3_aim = self.load_from_DRAM_multi_channel(self.w3.shape, self.w3_row_index, self.mode["weights"], self.dic_shape["w3"][0], False)
            x1_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w1_aim.T).reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w3_aim.T).reshape(bsz, 1, -1)
            self.dic_shape["x1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_row_index, self.mode["vector"], False)
            self.dic_shape["x3"] = self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x3_row_index, self.mode["vector"], False)
        compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
        compare(x3_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w3.T), "x3")

        # AiM AF EWMUL
        x1_sigmoid = torch.sigmoid(x1_aim)
        if self.pim_compute:
            # compare(x1_sigmoid_aim, torch.sigmoid(x1_aim), "x1 sigmoid")
            iteration_required = x1_aim.shape[-1] > self.channels_per_block * (self.num_banks // 4) * self.DRAM_column
            if iteration_required:
                iteration_0 = self.total_banks // 4 * 1024
                self.dic_shape["x1_bank_group_0"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.dic_shape["x1_bank_group_1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_1", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], False)
                x1_silu_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], False)
                x1_silu = torch.cat((x1_silu_0, x1_silu_1), dim=2)
            else:
                self.dic_shape["x1_bank_group"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], False)
            compare(x1_silu[0][0], (x1_aim * x1_sigmoid)[0][0], "x1_silu")
        else:
            compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
            self.dic_shape["x1_sigmoid"] = self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)
            x1_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_sigmoid_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu = self.Vector_Vector_EWMUL(x1_aim_load, x1_sigmoid_load)
            self.store_to_DRAM_multi_channel(x1_silu[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)

        # AiM EWMUL
        if self.pim_compute:
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    if iteration_required:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                    else:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
            if iteration_required:
                self.store_to_DRAM_multi_channel(x3_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x3_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], self.trace_activation)
                ffn_vector_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], self.trace_activation)
                ffn_vector = torch.cat((ffn_vector_0, ffn_vector_1), dim=2)
            else:
                self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], self.trace_activation)
        else:
            x3_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x3_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            ffn_vector = self.Vector_Vector_EWMUL(x1_silu_load, x3_aim_load)
        compare(ffn_vector[0][0], (F.silu(x1_aim) * x3_aim)[0][0], "ffn_vector")

        # AiM MAC BK x GB
        if self.pim_compute:
            ffn_aim = self.Vector_Matrix_Mul_weight_pim(ffn_vector[0][0], self.w2_row_index, self.w1.shape[0], self.w2.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight").reshape(bsz, 1, -1)
        else:
            w2_aim = self.load_from_DRAM_multi_channel(self.w2.shape, self.w2_row_index, self.mode["weights"], self.dic_shape["w2"][0], False)
            ffn_aim = self.Vector_Matrix_Mul_multithreads(ffn_vector[0][0], w2_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["ffn_bank_group"] = self.store_to_DRAM_multi_channel(ffn_aim[0][0], self.ffn_row_index, "vector_bank_group_1", False)
        compare(ffn_aim[0][0], self.ffn[0][0], "Vector_Matrix_Mul ffn")

        # AiM EWADD
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.ffn_row_index, "vector_bank_group_0", False)
        sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.ffn_row_index, "vector_bank_group_0", self.dic_shape["ffn_bank_group"][0], False)
        ffn_load = self.load_from_DRAM_multi_channel(self.ffn.shape, self.ffn_row_index, "vector_bank_group_1", self.dic_shape["ffn_bank_group"][0], False)
        out_aim = self.Vector_Vector_EWADD(sa_load, ffn_load)
        self.dic_shape["out_bank_group"] = self.store_to_DRAM_multi_channel(out_aim[0][0], self.ffn_row_index, "vector_bank_group_2", False)

        return out_aim
    
    def trace_only(self):
        bsz, _, _ = self.x.shape
        seqlen = self.seqlen
        total_banks = self.total_banks
        if self.model_parallel:
            FC_total_banks = total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = total_banks
            channels_required = self.channels_per_block
        channel_multi_transformer_block_required = self.num_channels // channels_required * channels_required
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)

        input_vector_neighbor_bank_length = (self.dim - 1) // (self.total_banks // 2) + 1
        input_vector_neighbor_bank_utilized_banks = (self.dim - 1) // input_vector_neighbor_bank_length + 1
        if self.trace_norm:
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 0, self.x_row_index, input_vector_neighbor_bank_length)
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 1, self.x_row_index, input_vector_neighbor_bank_length)

        # RMSNorm   x.pow   MAC_ABK
        input_vector_MAB_BK_BK_length = (self.dim - 1) // (total_banks // 2) + 1
        if self.trace_norm:
            self.WR_BIAS_only_trace(channel_lst)
            self.MAC_ABK_only_trace(channel_lst, self.x_row_index, (input_vector_MAB_BK_BK_length - 1) // self.burst_length + 1, "breakdown_sa_pow")
            self.RD_MAC_only_trace(channel_lst)

        # CXL Port  
        # Reduction of dim // 16 intermidiate sum read from MAC
        # Broadcast a scalar to vector and store it for EWMUL
        input_vector_EWMUL_length = (self.dim - 1) // (total_banks // 4) + 1
        input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1
        if self.trace_norm:
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.x_copy_row_index, input_vector_EWMUL_length)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.x_copy_row_index, input_vector_EWMUL_length)

            # RMSNorm   EWMUL
            self.EWMUL_only_trace(channel_lst, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            for bank in range(self.num_banks):
                bank_group_index = 2
                if bank % 4 == bank_group_index:
                    self.COPY_BK_GB_only_trace(channel_lst, bank, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
                    self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
            self.EWMUL_only_trace(channel_lst, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            # Read RMSNorm result vector to GPR
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim // self.burst_length
            self.load_from_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.SANorm_row_index, input_vector_EWMUL_length)
            self.SYNC_only_trace()

        # K/Q/V GEMV
        if self.trace_fc_kqvo:
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wq_row_index, self.dim, self.head_dim * self.n_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wk_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wv_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")

            # CXL Port
            # Store re-mapped xq/xk for EWMUL
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            # Rotary embedding
            self.EWMUL_only_trace(channel_lst, self.xq_row_index, self.dim // self.burst_length)
            self.EWMUL_only_trace(channel_lst, self.xk_row_index, self.dim // self.n_repeat // self.burst_length)
            # Load rotary embedding results
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)

        if self.trace_attention:
            # Store xk
            seq = seqlen - 1
            dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
            rows = self.head_dim * self.n_kv_heads // self.DRAM_column
            for row in range(rows):
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.DRAM_column // self.burst_length
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + tb * channels_required, bank_index, self.cache_k_row_index + seq // self.FC_total_banks * rows + row, self.DRAM_column)
            # Store xv
            if self.intra_device_attention:
                num_rows_per_seq = (seq - 1) // self.DRAM_column + 1
                row_offset = num_rows_per_seq - 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                dim_iteration = self.head_dim // self.num_banks
                for head_index_per_bank in range(num_heads_per_bank):
                    row_current_head = self.cache_v_row_index + (rows_per_dim * dim_iteration) * head_index_per_bank
                    for dim_iter in range(dim_iteration):
                        for channel in range(channels_required):
                            head = channel * num_heads_per_bank + head_index_per_bank
                            if head > self.n_kv_heads - 1:
                                break
                            # for bank in range(self.num_banks):
                            #     dim = dim_iter * self.num_banks + bank
                            #     self.W_MEM_only_trace(channel, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 1)
                            self.WR_ABK_only_trace(channel, row_current_head + dim_iter * rows_per_dim + row_offset, 1)
            else:
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                for channel in range(channels_required):
                    if banks_per_head < self.num_banks:
                        # print("banks_per_head", banks_per_head)
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            row_offset = rows_per_seq - 1
                            if self.trace_attention and channel < self.num_channels:
                                self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, 1)
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel % channels_per_head) * self.num_banks + bank
                                self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            if (channel % channels_per_head) // channels_per_row_offset == rows_per_seq - 1:
                                row_offset = rows_per_seq - 1
                                if self.trace_attention and channel < self.num_channels:
                                    self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset // rows_per_seq_iteration, 1)
                                for bank in range(self.num_banks):
                                    dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                                    self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset // rows_per_seq_iteration, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)

        else:
            cache_v_size = torch.Size([bsz, self.n_kv_heads, self.head_dim, -1])
            cache_k = self.load_from_DRAM_multi_channel(self.cache_k.shape, self.cache_k_row_index, self.mode["cache_k"], self.cache_k.shape[1], False)
            cache_v = self.load_from_DRAM_multi_channel(cache_v_size, self.cache_v_row_index, self.mode["cache_v"], self.cache_k.shape[1], False).transpose(2, 3).transpose(1, 2).reshape(bsz, -1, self.n_kv_heads, self.head_dim)
            compare(cache_k, self.cache_k, "cache v old")
            compare(cache_v, self.cache_v, "cache k old")
            cache_k[:bsz, self.start_pos : self.start_pos + 1] = xk_aim
            cache_v[:bsz, self.start_pos : self.start_pos + 1] = xv_aim

            keys_aim = cache_k[:bsz, : self.start_pos + 1]
            values_aim = cache_v[:bsz, : self.start_pos + 1]
            if self.GQA:
                keys_aim = repeat_kv(keys_aim, self.n_repeat)
                values_aim = repeat_kv(values_aim, self.n_repeat)
            xq_aim_load = xq_aim_load.transpose(1, 2)
            keys_aim = keys_aim.transpose(1, 2).transpose(2, 3)
            values_aim = values_aim.transpose(1, 2)

        # AiM MAC BK x GB
        if self.pim_compute:
            scores_aim = self.Vector_Matrix_Mul_score_pim(self.xq_row_index, self.cache_k_row_index, self.trace_attention, "breakdown_sa_score")

            if debug:
                self.cache_k[:bsz, self.start_pos : self.start_pos + seqlen] = xk_aim
                self.cache_v[:bsz, self.start_pos : self.start_pos + seqlen] = xv_aim
                keys = self.cache_k[:bsz, : self.start_pos + seqlen]
                values = self.cache_v[:bsz, : self.start_pos + seqlen]
                if self.GQA:
                    keys = repeat_kv(keys, self.n_repeat)
                    values = repeat_kv(values, self.n_repeat)
                xq = xq_aim.transpose(1, 2)
                keys = keys.transpose(1, 2).transpose(2, 3)
                compare(scores_aim, torch.matmul(xq, keys), "Vector_Matrix_Mul score")
        else:
            scores_aim = []
            for i in range(self.n_heads):
                scores_aim.append(self.Vector_Matrix_Mul(xq_aim_load[0][i][0], keys_aim[0][i], False))
            scores_aim = torch.tensor(scores_aim).reshape(bsz, self.n_heads, 1, -1)

        # CXL Ports
        head_dim_reciprocal = torch.full(scores_aim.shape, 1 / math.sqrt(self.head_dim))
        self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(head_dim_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in channel_lst:
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
        else:
            scores_aim_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            head_dim_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_aim_load, head_dim_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)

        # CXL Ports
        scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        if self.pim_compute and debug:
            scores = torch.matmul(xq, keys) / math.sqrt(self.head_dim)
            compare(scores_aim, scores, "Vector_Matrix_Mul score / head_dim")

        scores_exp = torch.exp(scores_aim)
        scores_exp_sum_reciprocal = 1 / torch.sum(scores_exp, dim=-1, keepdim=True)
        scores_exp_sum_reciprocal = torch.cat([scores_exp_sum_reciprocal for i in range(scores_exp.shape[-1])], dim=-1)
        self.store_to_DRAM_multi_channel(scores_exp.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(scores_exp_sum_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in range(channels_required):
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
            scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        else:
            scores_exp_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            scores_exp_sum_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_exp_load, scores_exp_sum_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)
        compare(scores_aim, self.scores, "SoftMax scores")

        # AiM MAC BK x GB
        if self.pim_compute:
            output_aim = self.Vector_Matrix_Mul_output_pim(scores_aim, self.cache_v_row_index, self.trace_attention, "breakdown_sa_output").reshape(bsz, 1, -1)
        else:
            output_aim = []
            for i in range(self.n_heads):
                output_aim.append(self.Vector_Matrix_Mul(scores_aim[0][i][0], values_aim[0][i], False))
            output_aim = torch.tensor(output_aim).reshape(bsz, 1, -1)
        compare(output_aim[0][0], self.output[0][0], "Vector_Matrix_Mul output")

        # CXL Ports
        self.dic_shape["output"] = self.store_to_DRAM_multi_channel(output_aim[0][0], self.output_row_index, self.mode["vector"], False)

        # AiM MAC BK x GB
        if self.pim_compute:
            sa_aim = self.Vector_Matrix_Mul_weight_pim(output_aim[0][0], self.wo_row_index, self.dim, self.wo.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
        else:
            wo_aim = self.load_from_DRAM_multi_channel(self.wo.shape, self.wo_row_index, self.mode["weights"], self.dic_shape["wo"][0], False)
            sa_aim = self.Vector_Matrix_Mul_multithreads(output_aim[0][0], wo_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_0", False)
        compare(sa_aim[0][0], self.sa[0][0], "Vector_Matrix_Mul sa")

        # CXL Ports
        sa_aim_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
        x_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
        sa_aim = self.Vector_Vector_EWADD(x_load, sa_aim_load)
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_2", False)

        return sa_aim
    
    def FFN(self, sa):
        compare(sa[0][0], self.h[0][0], "h")
        RMSNorm_sa = RMSNorm(sa, self.FFNNorm)
        x1 = F.linear(RMSNorm_sa, self.w1)
        x3 = F.linear(RMSNorm_sa, self.w3)
        ffn = F.linear(F.silu(x1) * x3, self.w2)
        compare(ffn[0][0], self.ffn[0][0], "ffn")
        out = sa + ffn
        return out
    
    def FFN_aim(self, sa_aim):
        bsz, _, _ = self.sa.shape
        if self.model_parallel:
            FC_total_banks = self.total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = self.total_banks
            channels_required = self.channels_per_block
        channel_lst = [channel for channel in range(channels_required)]

        # AiM MAC BK x BK
        if self.pim_compute:
            self.dic_shape["sa_neighbor_bank"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_0", self.trace_norm)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_1", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            sa_pow_sum = 0
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.WR_BIAS(0, channel, channels_required, 0, [0 for bank in range(self.num_banks)], op_trace)
                op_size = (self.dic_shape["sa_neighbor_bank"][0] - 1) // self.burst_length + 1
                self.MAC_BK_BK(0, channel, channels_required, self.sa_copy_row_index, 0, 0, op_size, op_trace)
                sa_pow_sum += sum(self.RD_MAC(0, channel, channels_required, 0, op_trace))    # CXL ports
        else:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", False)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_1", False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            sa_copy_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_pow_sum = self.Vector_Vector_Mul(sa_load[0][0], sa_copy_load[0][0], False)

        # CXL Ports
        compare(sa_pow_sum, sa_aim.pow(2).sum(), "sa pow")
        norm = torch.rsqrt(sa_pow_sum / self.dim + 1e-5)
        norm_tensor = torch.full(sa_aim.shape, norm)
        self.store_to_DRAM_multi_channel(norm_tensor[0][0], self.sa_copy_row_index, "vector_bank_group_1", self.trace_norm)
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length

        # AiM EWMUL
        if self.pim_compute:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.sa_copy_row_index, 0, op_size, op_trace)
        else:
            norm_tensor_load = self.load_from_DRAM_multi_channel(self.x.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            norm_sa = self.Vector_Vector_EWMUL(sa_load, norm_tensor_load)
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        # AiM EWMUL
        if self.pim_compute:
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_norm
                    self.COPY_BK_GB(0, channel, channels_required, bank, self.sa_copy_row_index, 0, op_size, op_trace)
                    self.COPY_GB_BK(0, channel, channels_required, bank-1, self.FFNNorm_row_index, 0, op_size, op_trace)
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.FFNNorm_row_index, 0, op_size, op_trace)
            FFNNorm_sa_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_2", self.dic_shape["FFNNorm"][0], self.trace_norm)
        else:
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_1", False)
            norm_sa_load = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            FFNNorm_load = self.load_from_DRAM_multi_channel(self.FFNNorm.shape, self.FFNNorm_row_index, "vector_bank_group_0", self.dic_shape["FFNNorm"][0], False)
            FFNNorm_sa_aim = self.Vector_Vector_EWMUL(norm_sa_load, FFNNorm_load)
            self.store_to_DRAM_multi_channel(FFNNorm_sa_aim[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        bsz, _, _ = FFNNorm_sa_aim.shape
        compare(FFNNorm_sa_aim[0][0], RMSNorm(sa_aim[0][0], self.FFNNorm), "FFNNorm_sa_aim")

        # AiM MAC BK x GB
        if self.pim_compute:
            x1_aim, x1_sigmoid_aim = self.Vector_Matrix_Mul_weight_af_pim(FFNNorm_sa_aim[0][0], self.w1_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")
            x1_aim = x1_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x1_sigmoid_aim = x1_sigmoid_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_weight_pim(FFNNorm_sa_aim[0][0], self.w3_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")[:self.w3.shape[0]].reshape(bsz, 1, -1)
        else:
            w1_aim = self.load_from_DRAM_multi_channel(self.w1.shape, self.w1_row_index, self.mode["weights"], self.dic_shape["w1"][0], False)
            w3_aim = self.load_from_DRAM_multi_channel(self.w3.shape, self.w3_row_index, self.mode["weights"], self.dic_shape["w3"][0], False)
            x1_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w1_aim.T).reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w3_aim.T).reshape(bsz, 1, -1)
            self.dic_shape["x1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_row_index, self.mode["vector"], False)
            self.dic_shape["x3"] = self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x3_row_index, self.mode["vector"], False)
        compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
        compare(x3_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w3.T), "x3")

        # AiM AF EWMUL
        x1_sigmoid = torch.sigmoid(x1_aim)
        if self.pim_compute:
            # compare(x1_sigmoid_aim, torch.sigmoid(x1_aim), "x1 sigmoid")
            iteration_required = x1_aim.shape[-1] > self.channels_per_block * (self.num_banks // 4) * self.DRAM_column
            if iteration_required:
                iteration_0 = self.total_banks // 4 * 1024
                self.dic_shape["x1_bank_group_0"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.dic_shape["x1_bank_group_1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_1", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], False)
                x1_silu_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], False)
                x1_silu = torch.cat((x1_silu_0, x1_silu_1), dim=2)
            else:
                self.dic_shape["x1_bank_group"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], False)
            compare(x1_silu[0][0], (x1_aim * x1_sigmoid)[0][0], "x1_silu")
        else:
            compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
            self.dic_shape["x1_sigmoid"] = self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)
            x1_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_sigmoid_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu = self.Vector_Vector_EWMUL(x1_aim_load, x1_sigmoid_load)
            self.store_to_DRAM_multi_channel(x1_silu[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)

        # AiM EWMUL
        if self.pim_compute:
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    if iteration_required:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                    else:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
            if iteration_required:
                self.store_to_DRAM_multi_channel(x3_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x3_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], self.trace_activation)
                ffn_vector_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], self.trace_activation)
                ffn_vector = torch.cat((ffn_vector_0, ffn_vector_1), dim=2)
            else:
                self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], self.trace_activation)
        else:
            x3_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x3_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            ffn_vector = self.Vector_Vector_EWMUL(x1_silu_load, x3_aim_load)
        compare(ffn_vector[0][0], (F.silu(x1_aim) * x3_aim)[0][0], "ffn_vector")

        # AiM MAC BK x GB
        if self.pim_compute:
            ffn_aim = self.Vector_Matrix_Mul_weight_pim(ffn_vector[0][0], self.w2_row_index, self.w1.shape[0], self.w2.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight").reshape(bsz, 1, -1)
        else:
            w2_aim = self.load_from_DRAM_multi_channel(self.w2.shape, self.w2_row_index, self.mode["weights"], self.dic_shape["w2"][0], False)
            ffn_aim = self.Vector_Matrix_Mul_multithreads(ffn_vector[0][0], w2_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["ffn_bank_group"] = self.store_to_DRAM_multi_channel(ffn_aim[0][0], self.ffn_row_index, "vector_bank_group_1", False)
        compare(ffn_aim[0][0], self.ffn[0][0], "Vector_Matrix_Mul ffn")

        # AiM EWADD
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.ffn_row_index, "vector_bank_group_0", False)
        sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.ffn_row_index, "vector_bank_group_0", self.dic_shape["ffn_bank_group"][0], False)
        ffn_load = self.load_from_DRAM_multi_channel(self.ffn.shape, self.ffn_row_index, "vector_bank_group_1", self.dic_shape["ffn_bank_group"][0], False)
        out_aim = self.Vector_Vector_EWADD(sa_load, ffn_load)
        self.dic_shape["out_bank_group"] = self.store_to_DRAM_multi_channel(out_aim[0][0], self.ffn_row_index, "vector_bank_group_2", False)

        return out_aim
    
    def trace_only(self):
        bsz, _, _ = self.x.shape
        seqlen = self.seqlen
        total_banks = self.total_banks
        if self.model_parallel:
            FC_total_banks = total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = total_banks
            channels_required = self.channels_per_block
        channel_multi_transformer_block_required = self.num_channels // channels_required * channels_required
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)

        input_vector_neighbor_bank_length = (self.dim - 1) // (self.total_banks // 2) + 1
        input_vector_neighbor_bank_utilized_banks = (self.dim - 1) // input_vector_neighbor_bank_length + 1
        if self.trace_norm:
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 0, self.x_row_index, input_vector_neighbor_bank_length)
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 1, self.x_row_index, input_vector_neighbor_bank_length)

        # RMSNorm   x.pow   MAC_ABK
        input_vector_MAB_BK_BK_length = (self.dim - 1) // (total_banks // 2) + 1
        if self.trace_norm:
            self.WR_BIAS_only_trace(channel_lst)
            self.MAC_ABK_only_trace(channel_lst, self.x_row_index, (input_vector_MAB_BK_BK_length - 1) // self.burst_length + 1, "breakdown_sa_pow")
            self.RD_MAC_only_trace(channel_lst)

        # CXL Port  
        # Reduction of dim // 16 intermidiate sum read from MAC
        # Broadcast a scalar to vector and store it for EWMUL
        input_vector_EWMUL_length = (self.dim - 1) // (total_banks // 4) + 1
        input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1
        if self.trace_norm:
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.x_copy_row_index, input_vector_EWMUL_length)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.x_copy_row_index, input_vector_EWMUL_length)

            # RMSNorm   EWMUL
            self.EWMUL_only_trace(channel_lst, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            for bank in range(self.num_banks):
                bank_group_index = 2
                if bank % 4 == bank_group_index:
                    self.COPY_BK_GB_only_trace(channel_lst, bank, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
                    self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
            self.EWMUL_only_trace(channel_lst, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            # Read RMSNorm result vector to GPR
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim // self.burst_length
            self.load_from_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.SANorm_row_index, input_vector_EWMUL_length)
            self.SYNC_only_trace()

        # K/Q/V GEMV
        if self.trace_fc_kqvo:
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wq_row_index, self.dim, self.head_dim * self.n_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wk_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wv_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")

            # CXL Port
            # Store re-mapped xq/xk for EWMUL
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            # Rotary embedding
            self.EWMUL_only_trace(channel_lst, self.xq_row_index, self.dim // self.burst_length)
            self.EWMUL_only_trace(channel_lst, self.xk_row_index, self.dim // self.n_repeat // self.burst_length)
            # Load rotary embedding results
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)

        if self.trace_attention:
            # Store xk
            seq = seqlen - 1
            dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
            rows = self.head_dim * self.n_kv_heads // self.DRAM_column
            for row in range(rows):
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.DRAM_column // self.burst_length
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + tb * channels_required, bank_index, self.cache_k_row_index + seq // self.FC_total_banks * rows + row, self.DRAM_column)
            # Store xv
            if self.intra_device_attention:
                num_rows_per_seq = (seq - 1) // self.DRAM_column + 1
                row_offset = num_rows_per_seq - 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                dim_iteration = self.head_dim // self.num_banks
                for head_index_per_bank in range(num_heads_per_bank):
                    row_current_head = self.cache_v_row_index + (rows_per_dim * dim_iteration) * head_index_per_bank
                    for dim_iter in range(dim_iteration):
                        for channel in range(channels_required):
                            head = channel * num_heads_per_bank + head_index_per_bank
                            if head > self.n_kv_heads - 1:
                                break
                            # for bank in range(self.num_banks):
                            #     dim = dim_iter * self.num_banks + bank
                            #     self.W_MEM_only_trace(channel, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 1)
                            self.WR_ABK_only_trace(channel, row_current_head + dim_iter * rows_per_dim + row_offset, 1)
            else:
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                for channel in range(channels_required):
                    if banks_per_head < self.num_banks:
                        # print("banks_per_head", banks_per_head)
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            row_offset = rows_per_seq - 1
                            if self.trace_attention and channel < self.num_channels:
                                self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, 1)
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel % channels_per_head) * self.num_banks + bank
                                self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            if (channel % channels_per_head) // channels_per_row_offset == rows_per_seq - 1:
                                row_offset = rows_per_seq - 1
                                if self.trace_attention and channel < self.num_channels:
                                    self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset // rows_per_seq_iteration, 1)
                                for bank in range(self.num_banks):
                                    dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                                    self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset // rows_per_seq_iteration, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)

        else:
            cache_v_size = torch.Size([bsz, self.n_kv_heads, self.head_dim, -1])
            cache_k = self.load_from_DRAM_multi_channel(self.cache_k.shape, self.cache_k_row_index, self.mode["cache_k"], self.cache_k.shape[1], False)
            cache_v = self.load_from_DRAM_multi_channel(cache_v_size, self.cache_v_row_index, self.mode["cache_v"], self.cache_k.shape[1], False).transpose(2, 3).transpose(1, 2).reshape(bsz, -1, self.n_kv_heads, self.head_dim)
            compare(cache_k, self.cache_k, "cache v old")
            compare(cache_v, self.cache_v, "cache k old")
            cache_k[:bsz, self.start_pos : self.start_pos + 1] = xk_aim
            cache_v[:bsz, self.start_pos : self.start_pos + 1] = xv_aim

            keys_aim = cache_k[:bsz, : self.start_pos + 1]
            values_aim = cache_v[:bsz, : self.start_pos + 1]
            if self.GQA:
                keys_aim = repeat_kv(keys_aim, self.n_repeat)
                values_aim = repeat_kv(values_aim, self.n_repeat)
            xq_aim_load = xq_aim_load.transpose(1, 2)
            keys_aim = keys_aim.transpose(1, 2).transpose(2, 3)
            values_aim = values_aim.transpose(1, 2)

        # AiM MAC BK x GB
        if self.pim_compute:
            scores_aim = self.Vector_Matrix_Mul_score_pim(self.xq_row_index, self.cache_k_row_index, self.trace_attention, "breakdown_sa_score")

            if debug:
                self.cache_k[:bsz, self.start_pos : self.start_pos + seqlen] = xk_aim
                self.cache_v[:bsz, self.start_pos : self.start_pos + seqlen] = xv_aim
                keys = self.cache_k[:bsz, : self.start_pos + seqlen]
                values = self.cache_v[:bsz, : self.start_pos + seqlen]
                if self.GQA:
                    keys = repeat_kv(keys, self.n_repeat)
                    values = repeat_kv(values, self.n_repeat)
                xq = xq_aim.transpose(1, 2)
                keys = keys.transpose(1, 2).transpose(2, 3)
                compare(scores_aim, torch.matmul(xq, keys), "Vector_Matrix_Mul score")
        else:
            scores_aim = []
            for i in range(self.n_heads):
                scores_aim.append(self.Vector_Matrix_Mul(xq_aim_load[0][i][0], keys_aim[0][i], False))
            scores_aim = torch.tensor(scores_aim).reshape(bsz, self.n_heads, 1, -1)

        # CXL Ports
        head_dim_reciprocal = torch.full(scores_aim.shape, 1 / math.sqrt(self.head_dim))
        self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(head_dim_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in channel_lst:
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
        else:
            scores_aim_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            head_dim_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_aim_load, head_dim_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)

        # CXL Ports
        scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        if self.pim_compute and debug:
            scores = torch.matmul(xq, keys) / math.sqrt(self.head_dim)
            compare(scores_aim, scores, "Vector_Matrix_Mul score / head_dim")

        scores_exp = torch.exp(scores_aim)
        scores_exp_sum_reciprocal = 1 / torch.sum(scores_exp, dim=-1, keepdim=True)
        scores_exp_sum_reciprocal = torch.cat([scores_exp_sum_reciprocal for i in range(scores_exp.shape[-1])], dim=-1)
        self.store_to_DRAM_multi_channel(scores_exp.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(scores_exp_sum_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in range(channels_required):
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
            scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        else:
            scores_exp_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            scores_exp_sum_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_exp_load, scores_exp_sum_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)
        compare(scores_aim, self.scores, "SoftMax scores")

        # AiM MAC BK x GB
        if self.pim_compute:
            output_aim = self.Vector_Matrix_Mul_output_pim(scores_aim, self.cache_v_row_index, self.trace_attention, "breakdown_sa_output").reshape(bsz, 1, -1)
        else:
            output_aim = []
            for i in range(self.n_heads):
                output_aim.append(self.Vector_Matrix_Mul(scores_aim[0][i][0], values_aim[0][i], False))
            output_aim = torch.tensor(output_aim).reshape(bsz, 1, -1)
        compare(output_aim[0][0], self.output[0][0], "Vector_Matrix_Mul output")

        # CXL Ports
        self.dic_shape["output"] = self.store_to_DRAM_multi_channel(output_aim[0][0], self.output_row_index, self.mode["vector"], False)

        # AiM MAC BK x GB
        if self.pim_compute:
            sa_aim = self.Vector_Matrix_Mul_weight_pim(output_aim[0][0], self.wo_row_index, self.dim, self.wo.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
        else:
            wo_aim = self.load_from_DRAM_multi_channel(self.wo.shape, self.wo_row_index, self.mode["weights"], self.dic_shape["wo"][0], False)
            sa_aim = self.Vector_Matrix_Mul_multithreads(output_aim[0][0], wo_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_0", False)
        compare(sa_aim[0][0], self.sa[0][0], "Vector_Matrix_Mul sa")

        # CXL Ports
        sa_aim_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
        x_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
        sa_aim = self.Vector_Vector_EWADD(x_load, sa_aim_load)
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_2", False)

        return sa_aim
    
    def FFN(self, sa):
        compare(sa[0][0], self.h[0][0], "h")
        RMSNorm_sa = RMSNorm(sa, self.FFNNorm)
        x1 = F.linear(RMSNorm_sa, self.w1)
        x3 = F.linear(RMSNorm_sa, self.w3)
        ffn = F.linear(F.silu(x1) * x3, self.w2)
        compare(ffn[0][0], self.ffn[0][0], "ffn")
        out = sa + ffn
        return out
    
    def FFN_aim(self, sa_aim):
        bsz, _, _ = self.sa.shape
        if self.model_parallel:
            FC_total_banks = self.total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = self.total_banks
            channels_required = self.channels_per_block
        channel_lst = [channel for channel in range(channels_required)]

        # AiM MAC BK x BK
        if self.pim_compute:
            self.dic_shape["sa_neighbor_bank"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_0", self.trace_norm)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_1", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            sa_pow_sum = 0
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.WR_BIAS(0, channel, channels_required, 0, [0 for bank in range(self.num_banks)], op_trace)
                op_size = (self.dic_shape["sa_neighbor_bank"][0] - 1) // self.burst_length + 1
                self.MAC_BK_BK(0, channel, channels_required, self.sa_copy_row_index, 0, 0, op_size, op_trace)
                sa_pow_sum += sum(self.RD_MAC(0, channel, channels_required, 0, op_trace))    # CXL ports
        else:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", False)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_1", False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            sa_copy_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_pow_sum = self.Vector_Vector_Mul(sa_load[0][0], sa_copy_load[0][0], False)

        # CXL Ports
        compare(sa_pow_sum, sa_aim.pow(2).sum(), "sa pow")
        norm = torch.rsqrt(sa_pow_sum / self.dim + 1e-5)
        norm_tensor = torch.full(sa_aim.shape, norm)
        self.store_to_DRAM_multi_channel(norm_tensor[0][0], self.sa_copy_row_index, "vector_bank_group_1", self.trace_norm)
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length

        # AiM EWMUL
        if self.pim_compute:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.sa_copy_row_index, 0, op_size, op_trace)
        else:
            norm_tensor_load = self.load_from_DRAM_multi_channel(self.x.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            norm_sa = self.Vector_Vector_EWMUL(sa_load, norm_tensor_load)
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        # AiM EWMUL
        if self.pim_compute:
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_norm
                    self.COPY_BK_GB(0, channel, channels_required, bank, self.sa_copy_row_index, 0, op_size, op_trace)
                    self.COPY_GB_BK(0, channel, channels_required, bank-1, self.FFNNorm_row_index, 0, op_size, op_trace)
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.FFNNorm_row_index, 0, op_size, op_trace)
            FFNNorm_sa_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_2", self.dic_shape["FFNNorm"][0], self.trace_norm)
        else:
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_1", False)
            norm_sa_load = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            FFNNorm_load = self.load_from_DRAM_multi_channel(self.FFNNorm.shape, self.FFNNorm_row_index, "vector_bank_group_0", self.dic_shape["FFNNorm"][0], False)
            FFNNorm_sa_aim = self.Vector_Vector_EWMUL(norm_sa_load, FFNNorm_load)
            self.store_to_DRAM_multi_channel(FFNNorm_sa_aim[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        bsz, _, _ = FFNNorm_sa_aim.shape
        compare(FFNNorm_sa_aim[0][0], RMSNorm(sa_aim[0][0], self.FFNNorm), "FFNNorm_sa_aim")

        # AiM MAC BK x GB
        if self.pim_compute:
            x1_aim, x1_sigmoid_aim = self.Vector_Matrix_Mul_weight_af_pim(FFNNorm_sa_aim[0][0], self.w1_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")
            x1_aim = x1_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x1_sigmoid_aim = x1_sigmoid_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_weight_pim(FFNNorm_sa_aim[0][0], self.w3_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")[:self.w3.shape[0]].reshape(bsz, 1, -1)
        else:
            w1_aim = self.load_from_DRAM_multi_channel(self.w1.shape, self.w1_row_index, self.mode["weights"], self.dic_shape["w1"][0], False)
            w3_aim = self.load_from_DRAM_multi_channel(self.w3.shape, self.w3_row_index, self.mode["weights"], self.dic_shape["w3"][0], False)
            x1_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w1_aim.T).reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w3_aim.T).reshape(bsz, 1, -1)
            self.dic_shape["x1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_row_index, self.mode["vector"], False)
            self.dic_shape["x3"] = self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x3_row_index, self.mode["vector"], False)
        compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
        compare(x3_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w3.T), "x3")

        # AiM AF EWMUL
        x1_sigmoid = torch.sigmoid(x1_aim)
        if self.pim_compute:
            # compare(x1_sigmoid_aim, torch.sigmoid(x1_aim), "x1 sigmoid")
            iteration_required = x1_aim.shape[-1] > self.channels_per_block * (self.num_banks // 4) * self.DRAM_column
            if iteration_required:
                iteration_0 = self.total_banks // 4 * 1024
                self.dic_shape["x1_bank_group_0"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.dic_shape["x1_bank_group_1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_1", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], False)
                x1_silu_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], False)
                x1_silu = torch.cat((x1_silu_0, x1_silu_1), dim=2)
            else:
                self.dic_shape["x1_bank_group"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], False)
            compare(x1_silu[0][0], (x1_aim * x1_sigmoid)[0][0], "x1_silu")
        else:
            compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
            self.dic_shape["x1_sigmoid"] = self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)
            x1_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_sigmoid_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu = self.Vector_Vector_EWMUL(x1_aim_load, x1_sigmoid_load)
            self.store_to_DRAM_multi_channel(x1_silu[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)

        # AiM EWMUL
        if self.pim_compute:
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    if iteration_required:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                    else:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
            if iteration_required:
                self.store_to_DRAM_multi_channel(x3_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x3_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], self.trace_activation)
                ffn_vector_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], self.trace_activation)
                ffn_vector = torch.cat((ffn_vector_0, ffn_vector_1), dim=2)
            else:
                self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], self.trace_activation)
        else:
            x3_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x3_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            ffn_vector = self.Vector_Vector_EWMUL(x1_silu_load, x3_aim_load)
        compare(ffn_vector[0][0], (F.silu(x1_aim) * x3_aim)[0][0], "ffn_vector")

        # AiM MAC BK x GB
        if self.pim_compute:
            ffn_aim = self.Vector_Matrix_Mul_weight_pim(ffn_vector[0][0], self.w2_row_index, self.w1.shape[0], self.w2.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight").reshape(bsz, 1, -1)
        else:
            w2_aim = self.load_from_DRAM_multi_channel(self.w2.shape, self.w2_row_index, self.mode["weights"], self.dic_shape["w2"][0], False)
            ffn_aim = self.Vector_Matrix_Mul_multithreads(ffn_vector[0][0], w2_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["ffn_bank_group"] = self.store_to_DRAM_multi_channel(ffn_aim[0][0], self.ffn_row_index, "vector_bank_group_1", False)
        compare(ffn_aim[0][0], self.ffn[0][0], "Vector_Matrix_Mul ffn")

        # AiM EWADD
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.ffn_row_index, "vector_bank_group_0", False)
        sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.ffn_row_index, "vector_bank_group_0", self.dic_shape["ffn_bank_group"][0], False)
        ffn_load = self.load_from_DRAM_multi_channel(self.ffn.shape, self.ffn_row_index, "vector_bank_group_1", self.dic_shape["ffn_bank_group"][0], False)
        out_aim = self.Vector_Vector_EWADD(sa_load, ffn_load)
        self.dic_shape["out_bank_group"] = self.store_to_DRAM_multi_channel(out_aim[0][0], self.ffn_row_index, "vector_bank_group_2", False)

        return out_aim
    
    def trace_only(self):
        bsz, _, _ = self.x.shape
        seqlen = self.seqlen
        total_banks = self.total_banks
        if self.model_parallel:
            FC_total_banks = total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = total_banks
            channels_required = self.channels_per_block
        channel_multi_transformer_block_required = self.num_channels // channels_required * channels_required
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)

        input_vector_neighbor_bank_length = (self.dim - 1) // (self.total_banks // 2) + 1
        input_vector_neighbor_bank_utilized_banks = (self.dim - 1) // input_vector_neighbor_bank_length + 1
        if self.trace_norm:
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 0, self.x_row_index, input_vector_neighbor_bank_length)
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 1, self.x_row_index, input_vector_neighbor_bank_length)

        # RMSNorm   x.pow   MAC_ABK
        input_vector_MAB_BK_BK_length = (self.dim - 1) // (total_banks // 2) + 1
        if self.trace_norm:
            self.WR_BIAS_only_trace(channel_lst)
            self.MAC_ABK_only_trace(channel_lst, self.x_row_index, (input_vector_MAB_BK_BK_length - 1) // self.burst_length + 1, "breakdown_sa_pow")
            self.RD_MAC_only_trace(channel_lst)

        # CXL Port  
        # Reduction of dim // 16 intermidiate sum read from MAC
        # Broadcast a scalar to vector and store it for EWMUL
        input_vector_EWMUL_length = (self.dim - 1) // (total_banks // 4) + 1
        input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1
        if self.trace_norm:
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.x_copy_row_index, input_vector_EWMUL_length)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.x_copy_row_index, input_vector_EWMUL_length)

            # RMSNorm   EWMUL
            self.EWMUL_only_trace(channel_lst, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            for bank in range(self.num_banks):
                bank_group_index = 2
                if bank % 4 == bank_group_index:
                    self.COPY_BK_GB_only_trace(channel_lst, bank, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
                    self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
            self.EWMUL_only_trace(channel_lst, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            # Read RMSNorm result vector to GPR
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim // self.burst_length
            self.load_from_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.SANorm_row_index, input_vector_EWMUL_length)
            self.SYNC_only_trace()

        # K/Q/V GEMV
        if self.trace_fc_kqvo:
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wq_row_index, self.dim, self.head_dim * self.n_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wk_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wv_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")

            # CXL Port
            # Store re-mapped xq/xk for EWMUL
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["WR_SBK"] += self.timing_constant["WR