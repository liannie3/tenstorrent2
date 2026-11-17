"""Verify hardware math first, then optionally verify the real prefill and cached decode."""
import argparse
import json


def model_check(path, d):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from decoder import Decoder
    hf = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager",
                                               local_files_only=True, use_safetensors=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    test = Decoder(hf, d)
    ids = tokenizer("The capital of Texas is", return_tensors="pt").input_ids
    previous = None
    for step in ("prefill", "cached_decode_1", "cached_decode_2"):
        expected = hf(input_ids=ids, use_cache=False).logits[0,-1].float()
        output = test(input_ids=ids if previous is None else ids[:,-1:], past_key_values=previous)
        actual = output.logits[0,-1]
        cosine = torch.nn.functional.cosine_similarity(expected, actual, dim=0).item()
        winner = int(expected.argmax())
        winners = actual.topk(5).indices.tolist()
        print(json.dumps(dict(stage=step, cosine=cosine, cpu_top1=winner, p150_top5=winners)))
        if not cosine >= 0.99 or winner not in winners:
            raise RuntimeError("Model check failed. Inspect precision, RoPE, head order and cache before timing.")
        previous = output.past_key_values
        ids = torch.cat((ids, torch.tensor([[winner]])), dim=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--model")
    args = p.parse_args()
    import ttnn as tt
    import torch
    d = tt.open_device(device_id=args.device)
    try:
        with torch.inference_mode():
            # A non-tile-aligned row count also checks logical shape handling.
            x = torch.arange(33*32).reshape(33,32).to(torch.bfloat16)
            def upload(a):
                return tt.from_torch(a, dtype=tt.bfloat16, layout=tt.TILE_LAYOUT, device=d)
            y = tt.matmul(upload(x), upload(torch.eye(32, dtype=torch.bfloat16)))
            tt.synchronize_device(d)
            torch.testing.assert_close(tt.to_torch(y).float(), x.float(), atol=1, rtol=0.01)
            print("PASS: TT-NN device matmul and host readback")
            if args.model:
                model_check(args.model, d)
                print("PASS: prefill and two teacher-forced cached decode steps")
    finally:
        tt.close_device(d)


if __name__ == "__main__":
    main()
