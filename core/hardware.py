import torch

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

class DynamicVRAMTracker:
    def __init__(self, force_cpu=False):
        self.force_cpu = force_cpu
        self.device = 'cpu' if force_cpu or not torch.cuda.is_available() else 'cuda'
        self.swap_count = 0
        
    def execute(self, func, *args, **kwargs):
        if self.force_cpu or not torch.cuda.is_available():
            return func(*args, device='cpu', **kwargs)

        if self.device == 'cpu':
            if HAS_GPUTIL:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus and gpus[0].memoryFree > 512:
                        self.device = 'cuda'
                except Exception:
                    pass
            else:
                try:
                    free_mem, _ = torch.cuda.mem_get_info()
                    if free_mem > 512 * 1024 * 1024:
                        self.device = 'cuda'
                except:
                    pass

        if self.device == 'cuda':
            try:
                return func(*args, device='cuda', **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    torch.cuda.empty_cache()
                    self.device = 'cpu'
                    self.swap_count += 1
                    return func(*args, device='cpu', **kwargs)
                else:
                    raise e
        else:
            return func(*args, device='cpu', **kwargs)