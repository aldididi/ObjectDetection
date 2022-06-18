# SKRIPSI

```18th June```
Still got an error ->

Error on request:
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/werkzeug/serving.py", line 335, in run_wsgi
    execute(self.server.app)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/werkzeug/serving.py", line 324, in execute
    for data in application_iter:
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/werkzeug/wsgi.py", line 462, in __next__
    return self._next()
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/werkzeug/wrappers/response.py", line 50, in _iter_encoded
    for item in iterable:
  File "/Users/aldymochamadheryana/Documents/dev/skripsi/web-detect-true/app.py", line 52, in gen
    model = model_load()
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/torch/autograd/grad_mode.py", line 27, in decorate_context
    return func(*args, **kwargs)
  File "/Users/aldymochamadheryana/Documents/dev/skripsi/web-detect-true/app.py", line 35, in model_load
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
  File "/Users/aldymochamadheryana/Documents/dev/skripsi/web-detect-true/models/common.py", line 307, in __init__
    model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
  File "/Users/aldymochamadheryana/Documents/dev/skripsi/web-detect-true/models/experimental.py", line 96, in attempt_load
    ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/torch/serialization.py", line 699, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/torch/serialization.py", line 231, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/torch/serialization.py", line 212, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'last.pt'
