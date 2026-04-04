import torch, sys
sys.path.insert(0, ".")
from guided_diffusion.script_util import create_model
from guided_diffusion.wunet import WaveletDownsample, TimestepBlock
from guided_diffusion.nn import timestep_embedding

model = create_model(
    image_size=96, num_channels=64, num_res_blocks=2, channel_mult="1,2,4,4",
    in_channels=32, out_channels=8, dims=3, num_heads=1, num_groups=32,
    attention_resolutions="12,6", bottleneck_attention=True, resample_2d=False,
    additive_skips=False, use_freq=True, use_cross_attn=True)
model.to("cuda").eval()
x = torch.randn(1, 32, 96, 96, 96, device="cuda")
t = torch.tensor([500], device="cuda")
with torch.no_grad():
    hs = []
    input_pyramid = x
    emb = model.time_embed(timestep_embedding(t, model.model_channels))
    h = x
    for i, module in enumerate(model.input_blocks):
        if not isinstance(module[0], WaveletDownsample):
            h = module(h, emb)
            skip = None; 
            if isinstance(h, tuple): h, skip = h
            hs.append(skip)
        else:
            input_pyramid = module(input_pyramid, emb)
            input_pyramid = input_pyramid + h
            h = input_pyramid
    for mod in model.middle_block:
        if isinstance(mod, TimestepBlock):
            h = mod(h, emb)
        else:
            h = mod(h)
        if isinstance(h, tuple): h, _ = h
        
    for i, module in enumerate(model.output_blocks):
        new_hs = hs.pop()
        if isinstance(h, tuple):
            l = list(h); l[1] = new_hs; h = tuple(l)
        else:
            h = (h, new_hs)
        
        if i == 2:
            print("=== DECODER BLOCK 2 TRACE ===")
            print(f"  initially h type: {type(h)}")
            if isinstance(h, tuple):
                print(f"  tuple length: {len(h)}")
                print(f"  h[0] type: {type(h[0])}, shape: {h[0].shape}")
            for j, sub in enumerate(module):
                print(f"  sub[{j}]: {type(sub).__name__}")
                try:
                    if isinstance(sub, TimestepBlock):
                        h = sub(h, emb)
                    else:
                        h = sub(h)
                    
                    print(f"    -> returned type: {type(h)}")
                    if isinstance(h, tuple):
                        print(f"    -> len: {len(h)}")
                        print(f"    -> h[0] type: {type(h[0])}")
                        if isinstance(h[0], torch.Tensor):
                            print(f"    -> h0 shape: {h[0].shape}")
                        elif isinstance(h[0], tuple):
                            print(f"    -> h[0] is tuple of len {len(h[0])}! Its element 0 shape is {h[0][0].shape}")
                    elif isinstance(h, torch.Tensor):
                        print(f"    -> shape: {h.shape}")
                except Exception as e:
                    print(f"    -> CRASH on sub[{j}]: {e}")
                    break
            break
        else:
            h = module(h, emb)
