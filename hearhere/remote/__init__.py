"""Remote backend: run the pipeline on a GPU worker (e.g. RunPod).

:mod:`~hearhere.remote.worker` serves the pipeline over HTTP;
:mod:`~hearhere.remote.client` uploads audio and downloads results;
:mod:`~hearhere.remote.protocol` is the shared contract. Audio leaves the
machine only via this backend — see :mod:`~hearhere.remote.consent`.
"""
