import torch
from fvcore.nn import FlopCountAnalysis

from main_space.model import MyTransformerLM


def get_other_model_params(model, counted_param_ids):

    other_params = 0
    other_params_details = {}

    for name, param in model.named_parameters():
        if id(param) not in counted_param_ids:
            other_params += param.numel()
            other_params_details[name] = param.numel()
            counted_param_ids.add(id(param))

    return other_params, other_params_details

def get_model_parameters(model):

    all_params = {}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tok_embd_params = 0
    pos_embd_params = 0
    non_embd_params = 0
    lm_head_params = 0

    counted_param_ids = set()

    if hasattr(model, 'token_embedding') and hasattr(model.token_embedding, 'parameters'):
        for p in model.token_embedding.parameters():
            if id(p) not in counted_param_ids:
                tok_embd_params += p.numel()
                counted_param_ids.add(id(p))

    if hasattr(model, 'positional_embedding') and hasattr(model.positional_embedding, 'parameters'):
        for p in model.positional_embedding.parameters():
            if id(p) not in counted_param_ids:
                pos_embd_params += p.numel()
                counted_param_ids.add(id(p))

    if hasattr(model, 'transformer_blocks') and hasattr(model.transformer_blocks, 'parameters'):
        for p in model.transformer_blocks.parameters():
            if id(p) not in counted_param_ids:
                non_embd_params += p.numel()
                counted_param_ids.add(id(p))

    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'parameters'):
        for p in model.lm_head.parameters():
            if id(p) not in counted_param_ids:
                lm_head_params += p.numel()
                counted_param_ids.add(id(p))


    if tok_embd_params + pos_embd_params + non_embd_params + lm_head_params != total_params:
        other_params, other_params_details = get_other_model_params(model, counted_param_ids)

        print("Token + pos + non_embd + lm_head params != total_params.\n"
                         f"total_params = {total_params}\n"
                         f"{tok_embd_params} + {pos_embd_params} + {non_embd_params} + {lm_head_params} = {tok_embd_params + pos_embd_params + non_embd_params + lm_head_params}\n"
                         f"other_params = {other_params}\n"
                         f"Other params details: \n{other_params_details}")
        print("Quitting script..")
        quit()


    all_params['total_params'] = total_params
    all_params['trainable_params'] = trainable_params

    all_params['tok_embd_params'] = tok_embd_params
    all_params['pos_embd_params'] = pos_embd_params
    all_params['non_embd_params'] = non_embd_params
    all_params['lm_head_params'] = lm_head_params

    return all_params



def calc_model_operation(model_cls, model_args, print_unsupported = False):
    operations = {}

    device = dummy_input_ids.device
    model = model_cls(**model_args).to(device)
    model.eval()  # IMPORTANT: Ensure model is in eval mode (e.g., dropouts are off, etc.)

    # print(flops_analyzer.by_module())  # for each nn.Module separately

    flops_analyzer = FlopCountAnalysis(model, dummy_input_ids)

    total_flops = flops_analyzer.total()
    embd_flops = total_flops - flops_analyzer.by_module()['transformer_blocks']
    non_embd_flops = flops_analyzer.by_module()['transformer_blocks']

    linear_flops = flops_analyzer.by_operator()['linear']
    matmul_flops = flops_analyzer.by_operator()['matmul']
    layer_norm_flops = flops_analyzer.by_operator()['layer_norm']

    operations['total_flops'] = total_flops
    operations['embd_flops'] = embd_flops
    operations['non_embd_flops'] = non_embd_flops

    operations['linear_flops'] = linear_flops
    operations['matmul_flops'] = matmul_flops
    operations['layer_norm_flops'] = layer_norm_flops


    if print_unsupported:
        unsupported_ops = flops_analyzer.unsupported_ops()
        if unsupported_ops:
            print("\nUnsupported operators by FLOPs counter (these are not included in the total):")
            for op_name, op_info in unsupported_ops.items():  # Iterate through dict items
                print(f"- {op_name}: {op_info} occurrence(s)")
        else:
            print("\nAll operators were supported by the FLOPs counter.")

    return operations


def calc_and_print_operations(model_cls, model_args, input_ids, unit):
    if unit not in OPS_UNIT_TYPES:
        raise ValueError(
            f"Parameter 'unit' should be one of {OPS_UNIT_TYPES}. {unit} was given to function calc_and_print_operations.")

    if unit == "MFLOPs":
        unit_divider = 1e6
    elif unit == "GFLOPs":
        unit_divider = 1e9
    elif unit == "TFLOPs":
        unit_divider = 1e12
    elif unit == "PFLOPs":
        unit_divider = 1e15
    elif unit == "PF-days":
        unit_divider = 1e15 * 60 * 60 * 24

    operations = calc_model_operation(model_cls, model_args)

    total_flops = operations['total_flops']
    embd_flops = operations['embd_flops']
    non_embd_flops = operations['non_embd_flops']

    print("-" * 64)
    print(f"Total FLOPs: {total_flops:_} ({total_flops / unit_divider:.2f} {unit})")
    print("-" * 64)
    print(f"Embedding FLOPs: {embd_flops:_} ({embd_flops / unit_divider:.2f} {unit})")
    print(f"Non-embedding FLOPs: {non_embd_flops:_} ({non_embd_flops / unit_divider:.2f} {unit})")
    print("-" * 64)

    linear_flops = operations['linear_flops']
    matmul_flops = operations['matmul_flops']
    layer_norm_flops = operations['layer_norm_flops']

    print(f"Linear layers FLOPs: {linear_flops:_} ({linear_flops / unit_divider:.2f} {unit})")
    print(f"Matmul FLOPs: {matmul_flops:_} ({matmul_flops / unit_divider:.2f} {unit})")
    print(f"Layer norm FLOPs: {layer_norm_flops:_} ({layer_norm_flops / unit_divider:.2f} {unit})")
    print("-" * 64)


def calc_and_print_params(model_cls, model_args, device, unit):
    if unit not in PARAMS_UNIT_TYPES:
        raise ValueError(
            f"Parameter 'unit' should be one of {PARAMS_UNIT_TYPES}. {unit} was given to function calc_and_print_params.")

    if unit == "":
        unit_divider = 1
    elif unit == "K":
        unit_divider = 1e3
    elif unit == "M":
        unit_divider = 1e6
    elif unit == "B":
        unit_divider = 1e9

    model = model_cls(**model_args).to(device)
    model.eval()

    all_params = get_model_parameters(model)
    total_params = all_params['total_params']

    trainable_params = all_params['trainable_params']
    non_trainable_params = total_params - trainable_params

    tok_embd_params = all_params['tok_embd_params']
    pos_embd_params = all_params['pos_embd_params']
    non_embd_params = all_params['non_embd_params']
    lm_head_params = all_params['lm_head_params']

    print(f"Total parameters: {total_params:_} ({total_params / unit_divider:.2f} {unit}) params")
    print("-" * 64)
    print(f"Trainable parameters: {trainable_params:_} ({trainable_params / unit_divider:.2f} {unit}) params")
    print(f"Non-trainable parameters: {non_trainable_params:_} ({non_trainable_params / unit_divider:.2f} {unit}) params")
    print("-" * 64)
    print(f"Token embedding parameters: {tok_embd_params:_} ({tok_embd_params / unit_divider:.2f} {unit}) params")
    print(f"Position embedding parameters: {pos_embd_params:_} ({pos_embd_params / unit_divider:.2f} {unit}) params")
    print(f"Non-embedding parameters: {non_embd_params:_} ({non_embd_params / unit_divider:.2f} {unit}) params")
    print(f"Lm head parameters: {lm_head_params:_} ({lm_head_params / unit_divider:.2f} {unit}) params")
    print("-" * 64)




if __name__ == '__main__':

    print_ops = False
    print_params = True

    OPS_UNIT_TYPES = ['MFLOPs', 'GFLOPs', 'TFLOPs', 'PFLOPs', 'PF-days']
    PARAMS_UNIT_TYPES = ['', 'K', 'M', 'B']

    unit_ops = 'GFLOPs'
    unit_params = 'M'

    vocab_size = 20000
    d_model = 256
    n_heads = 4
    n_layers = 3
    ctx_size = 1024
    p_dropout = 0.0

    MODEL_NAME = "MyTransformerLM"

    model_constructor_args = {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'ctx_size': ctx_size,
        'p_dropout': p_dropout
    }

    print('-' * 64)
    print(f"Model configuration:")
    for key, value in model_constructor_args.items():
        print(f"  {key}: {value}")

    batch_size_profile = 1
    seq_len_profile = ctx_size

    operating_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"\nUsing device: {operating_device}")

    dummy_input_ids = torch.randint(
        0, vocab_size,
        (batch_size_profile, seq_len_profile),
        device=operating_device,
        dtype=torch.long
    )

    if print_ops:
        print('-' * 64)
        calc_and_print_operations(
            model_cls=MyTransformerLM,
            model_args=model_constructor_args,
            input_ids=dummy_input_ids,
            unit=unit_ops
        )

    if print_params:
        if not print_ops:
            print("-" * 64)
        calc_and_print_params(model_cls=MyTransformerLM,
                              model_args=model_constructor_args,
                              device=operating_device,
                              unit=unit_params)
