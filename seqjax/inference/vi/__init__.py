# import seqjax.inference.vi.transformations
# import seqjax.inference.vi.autoregressive
# import seqjax.inference.vi.transformed
# import seqjax.inference.vi.train
# import seqjax.inference.vi.base
# import seqjax.inference.vi.embedder


from seqjax.inference import embedder

from . import train
from . import autoregressive
from . import base
from . import transformed
from . import transformations

from .run import run_buffered_vi, BufferedVIConfig, run_full_path_vi, FullVIConfig
