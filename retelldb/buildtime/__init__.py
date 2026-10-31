"""
Build-time only code.

Nothing in this package may be imported by the runtime retrieval path: it exists to
generate configuration (taxonomies, datasets) while a release is being built, and it
talks to paid model providers over the network. `retelldb` never imports it, and
`assert_build_time` refuses to run inside a production deployment.
"""

from retelldb.buildtime.adaption import AdaptionTransport, assert_build_time

__all__ = ["AdaptionTransport", "assert_build_time"]
