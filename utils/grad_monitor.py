def detect_abnormal_grad(model, grad_threshold):
    for name, param in model.named_parameters():
        print(f'name: {name}, param.grad is not None: {param.grad is not None}')
        if param.grad is not None:
            if param.grad.data.abs().max() > grad_threshold:
                print(f"Abnormal gradient detected in {name}: max abs value = {param.grad.data.abs().max()}")
            else:
                print(f"Normal gradient detected in {name}: max abs value = {param.grad.data.abs().max()}")