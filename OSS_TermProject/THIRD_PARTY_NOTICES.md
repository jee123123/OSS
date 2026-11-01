# Third-Party Notices

Score2MIDI uses or documents the following third-party software. These projects
retain their own licenses; this repository does not relicense them.

| Component | License | How it is used | Project |
|---|---|---|---|
| Audiveris | GNU AGPL-3.0 | The current command-line workflow invokes a separately installed executable. A future Windows desktop distribution is planned to include Audiveris as a separate program. | <https://github.com/Audiveris/audiveris> |
| music21 | BSD-3-Clause | Reads MusicXML and writes MIDI. | <https://github.com/cuthbertLab/music21> |
| PySide6 / Qt for Python | LGPL-3.0 and GPL-3.0 (module-dependent); commercial license also available | Optional dependency for the desktop GUI. | <https://doc.qt.io/qtforpython-6/> |
| OpenCV (`opencv-python`) | Apache-2.0 | Used by the legacy image-processing code. | <https://github.com/opencv/opencv> |
| NumPy | BSD-3-Clause | Used by the legacy image-processing code and OpenCV. | <https://github.com/numpy/numpy> |
| Pillow | MIT-CMU | Image loading and processing dependency. | <https://github.com/python-pillow/Pillow> |

See each project for its complete license text and any notices that apply to a
particular release or binary distribution. No Audiveris binary is included in
the current repository or command-line release. Before distributing an
installer that includes Audiveris, record the exact Audiveris version, provide
the corresponding source code and license notices, and identify it as a
separate program. Audiveris remains under its own license; this project does
not relicense it.
