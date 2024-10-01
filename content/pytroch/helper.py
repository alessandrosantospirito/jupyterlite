def save_params_anonymous(func):
    def wrapper(*args, **kwargs):
        prefix = "_"
        globals().update({prefix + k: v for k, v in zip(func.__code__.co_varnames, args)})
        return func(*args, **kwargs)
    return wrapper

def save_params(func):
    def wrapper(*args, **kwargs):
        globals().update({k: v for k, v in zip(func.__code__.co_varnames, args)})
        return func(*args, **kwargs)
    return wrapper
