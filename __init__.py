is_simple_core = True

if is_simple_core:
    from shuzPy.core_simple import Variable
    from shuzPy.core_simple import Function
    from shuzPy.core_simple import using_config
    from shuzPy.core_simple import no_grad
    from shuzPy.core_simple import as_array
    from shuzPy.core_simple import as_variable
    from shuzPy.core_simple import setup_variable

"""
else:
    from shuzPy.core import Variable
    from shuzPy.core import Function
    from shuzPy.core import using_config
    from shuzPy.core import no_grad
    from shuzPy.core import as_array
    from shuzPy.core import as_variable
    from shuzPy.core import setup_variable
"""
    
setup_variable()