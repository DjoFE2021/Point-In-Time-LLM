# Third-party notices

Parts of this repository are adapted from the MIT-licensed projects below. Their copyright notices and the MIT license text are reproduced here as the license requires.

## modded-nanogpt

- Source: https://github.com/KellerJordan/modded-nanogpt
- Used in: `models/GPT.py`, `optimizers/MuON.py`, `train_gpt.py`, `dataloaders/DDP.py`
- Copyright (c) 2024 Keller Jordan

## llm.c

- Source: https://github.com/karpathy/llm.c
- Used in: the `.bin` token-shard format and its reader/writer (`dataloaders/DDP.py`, `data/get_train_set.py`)
- Copyright (c) 2024 Andrej Karpathy

## ChronoGPT

- Source: https://huggingface.co/manelalab (`ChronoGPT_inference.py`, `ChronoGPT_instruct.py`)
- Used in: `models/ChronoGPTLMInstruct.py`, `eval/eval_chrono.py`
- Authors: Songrun He, Linying Lv, Asaf Manela, Jimmy Wu. The model cards state that it is released under the MIT License.

## MIT License (applies to each project above)

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
