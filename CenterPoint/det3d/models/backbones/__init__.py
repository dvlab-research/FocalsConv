import importlib
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None

if found:
    from .scn import SpMiddleResNetFHD
    from .scn_focal import SpMiddleResNetFHDFocal
    from .scn_largekernel_multimodal import SpMiddleResNetFHDLargeKernel
else:
    print("No spconv, sparse convolution disabled!")

from .resnet import *