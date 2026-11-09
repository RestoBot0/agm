# AGM V1 — AGM-L and AGM-P

AGM V1 is the first release of the AGM image codec. It has two encoding modes provided by the same C implementation and Python interface:

- **AGM-L** is lossless for supported 8-bit grayscale or RGB pixel data.
- **AGM-P** is perceptual and lossy. Its quality setting is an encoder control, not a guarantee of a particular file size or visual score.

Both modes produce AGM bitstreams with the `.agm` extension. The stream header identifies the mode: `AGML` for AGM-L and `AGMP` for AGM-P.

## Files

Keep the three V1 source files together in `VERSIONS/V1/`:

- `agm_codec.c` — C implementation of both AGM-L and AGM-P.
- `agm_codec.h` — C API declarations for both modes.
- `agm_codec.py` — Python command-line interface and SDK.

AGM-L and AGM-P do not have separate V1 C implementations: they share `agm_codec.c` and `agm_codec.h`.

## Requirements and setup

On Debian, Ubuntu, or Chromebook Linux, install the compiler and Python dependencies:

```sh
sudo apt update
sudo apt install -y gcc python3 python3-numpy python3-pil
```

Open a terminal in `VERSIONS/V1/`. The Python interface compiles the C implementation automatically the first time it is used. GCC must therefore be installed, and the C source must remain beside the Python file.

## Command-line examples

The quality argument selects the mode. If it is omitted, the default is `100` (AGM-L).

Encode losslessly with AGM-L:

```sh
python3 agm_codec.py encode input.png output.agm --quality 100
```

Encode perceptually with AGM-P (75 is an example setting):

```sh
python3 agm_codec.py encode input.png output.agm --quality 75
```

Decode either mode to an image file:

```sh
python3 agm_codec.py decode output.agm reconstructed.png
```

Inspect the mode and dimensions recorded in a bitstream:

```sh
python3 agm_codec.py info output.agm
```

Use `100` for AGM-L and a value from `1` through `99` for AGM-P. A lower or higher quality number does not guarantee a fixed bitrate, file size, or visual result; test the settings on the material you intend to encode.

## Python API example

```python
from PIL import Image
import numpy as np
import agm_codec

image = np.array(Image.open("input.png").convert("RGB"), dtype=np.uint8)

# AGM-L: the decoded RGB samples must match the samples given to the encoder.
lossless_data = agm_codec.encode(image, quality=100)
lossless_image = agm_codec.decode(lossless_data)
assert np.array_equal(image, lossless_image)

# AGM-P: lossy reconstruction. Evaluate visual quality for your application.
perceptual_data = agm_codec.encode(image, quality=75)
perceptual_image = agm_codec.decode(perceptual_data)
```

## C API

Include `agm_codec.h` and compile `agm_codec.c` with GCC, linking the math library. The unified encoder chooses the mode from `quality`:

```c
size_t agm_encode(
    const uint8_t *img, int w, int h, int channels,
    int quality, uint8_t *out_buf, size_t out_cap
);
```

Pass `100` for AGM-L or `1`–`99` for AGM-P. The function returns the encoded byte count; zero indicates failure. The header also provides mode-specific encode/decode functions. Check return values and provide valid dimensions and enough output-buffer capacity.

## Input and transparency

The codec handles 8-bit grayscale and RGB samples. The Python interface reads common Pillow-supported formats such as PNG, JPEG, and WebP. Images in other modes are converted to RGB before encoding. If a four-channel NumPy array is supplied, the current wrapper drops its fourth channel.

**V1 does not preserve transparency.** If alpha matters, store it separately. AGM-L guarantees exact reconstruction of the supported pixel samples passed to the codec; it does not reproduce the original file bytes, metadata, or color profile.

## Important notes

- AGM-P is lossy; inspect reconstructed images and measure quality on your own corpus.
- AGM bitstreams require an AGM decoder; changing the filename extension does not turn one into a PNG, JPEG, or WebP.
