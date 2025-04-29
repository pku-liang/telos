AGG_I_M = 0b1
OUT_I_M = 0b10
AGG_J_M = 0b100
OUT_J_M = 0b1000

AGG_I_INV_M = 0b10000
OUT_I_INV_M = 0b100000
AGG_J_INV_M = 0b1000000
OUT_J_INV_M = 0b10000000

CUR_M = 0b100000000

# Calculate the prefix OR of the masks
def gen_mask(mask_list):
    val = 0
    out_list = []
    for mask in mask_list:
        val |= mask
        out_list.append(val)
    return out_list