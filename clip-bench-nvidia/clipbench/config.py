import yaml
import torch

## Loads .yaml file and calls run_type function
def load_config(config_path='config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return determine_run_type(config)

## Determines if runtype is "cuda" or "cpu"
def determine_run_type(config):
    run = config['run']                                                                       
    have_cuda = torch.cuda.is_available()                                                    # Checks if platform is using cuda (Boolean)                         
                                                                                     
    requested = run['device']                                                                # Returns device type from .yaml             
    if requested == 'auto' and have_cuda:                                                    # "cuda" if True                         
        run['device'] = 'cuda'                                                                             
    elif requested == 'auto' and not have_cuda:                                              # "cpu" if True                               
        run['device'] = 'cpu'                                                                             
    elif requested == 'cuda' and not have_cuda:                                              # Error thrown if True                                                                
        raise RuntimeError("Requested == CUDA but torch.cuda.is_available() returns False")
                                                                                     
                                                                             
    if run["precision"] == "auto":                                                                             
        run["precision"] = "fp16" if run["device"] == "cuda" else "fp32"                                                                             
                                                                                

    return config                                                                             
    
