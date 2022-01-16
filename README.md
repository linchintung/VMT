# VMT

Video-Music Transformer (VMT) is an attention-based multi-modal model, which generates piano music for a given video.

## Paper

https://arxiv.org/abs/2112.15320

## Demo

Here are 5 selected 5 video fragments from our dataset.
Note that we do not do any post-production.
Each file is made from the original video with a WAVE file converted from the MIDI of the model output.

| ID      | Original                            | VMT                            | Seq2Seq                            |
| ------- | ----------------------------------- | ------------------------------ | ---------------------------------- |
| 100-001 | ![](docs/data/100-001_original.mp4) | ![](docs/data/100-001_vmt.mp4) | ![](docs/data/100-001_seq2seq.mp4) |
| 101-004 | ![](docs/data/101-004_original.mp4) | ![](docs/data/101-004_vmt.mp4) | ![](docs/data/101-004_seq2seq.mp4) |
| 115-005 | ![](docs/data/115-005_original.mp4) | ![](docs/data/115-005_vmt.mp4) | ![](docs/data/115-005_seq2seq.mp4) |
| 118-005 | ![](docs/data/118-005_original.mp4) | ![](docs/data/118-005_vmt.mp4) | ![](docs/data/118-005_seq2seq.mp4) |
| 122-005 | ![](docs/data/122-005_original.mp4) | ![](docs/data/122-005_vmt.mp4) | ![](docs/data/122-005_seq2seq.mp4) |
