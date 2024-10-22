from IPython.display import display, HTML

def html(function_name):
    display(HTML(function_name.__html__))

