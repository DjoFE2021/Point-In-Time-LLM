import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from lm_eval.api.model import TemplateLM

from torch.nn.utils.rnn import pad_sequence

def _chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

class PIT(TemplateLM):
    def __init__(self, model, tokenizer, max_length=None, batch_size=128):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size  # Used in _loglikelihood_tokens
        # Determine max context length (use model/tokenizer info or default)
        if max_length is not None:
            self.max_length = max_length
        else:
            # Try to get model context length from tokenizer or model config, else default
            self.max_length = getattr(tokenizer, "model_max_length", 2048)

        # Move model to eval mode and ensure no grad (we're evaluating)
        self.model.eval()

        # If using multiple devices or DataParallel, you might handle that here.
        # For simplicity, assume model is on a single device:
        self._device = next(self.model.parameters()).device

    @property
    def eot_token_id(self):
        # End-of-text token ID (often same as EOS token in GPT-like tokenizers)
        eos = None
        if hasattr(self.tokenizer, "eos_token_id"):
            eos = self.tokenizer.eos_token_id
        if eos is None and hasattr(self.tokenizer, "vocab"):
            # Fallback: last vocab index (for GPT2, e.g., 50256)
            eos = len(self.tokenizer) - 1
        return eos

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        # Tokenize the string using the provided tokenizer.
        # Avoid adding special tokens since harness manages BOS/EOS separately.
        if hasattr(self.tokenizer, "encode"):
            # HuggingFace tokenizers
            return self.tokenizer.encode(string, add_special_tokens=False)
        else:
            # e.g., Fast tokenizers library or custom
            return self.tokenizer.tokenize(string)  # adjust if needed for your tokenizer

    @torch.inference_mode()
    def _loglikelihood_tokens(self, requests, disable_tqdm=True):
        """
        Batched log-likelihood for (context, continuation) pairs.
        Returns: List[(logprob_sum, is_greedy)] in same order as `requests`.
        """
        results = []
        batch_size = getattr(self, "batch_size", 128)
        pad_id = self.eot_token_id if self.eot_token_id is not None else 0
        dev = self._device

        req_iter = _chunked(requests, batch_size)

        for batch in tqdm(req_iter, disable=disable_tqdm):
            # store original order results for this batch
            batch_out = []

            # ---- collect, fix spacing, truncate ----
            ctx_tensors, cont_tensors, metas = [], [], []
            for (ctx_str, cont_str), ctx_ids, cont_ids in batch:
                # Guarantee a boundary
                if ctx_str and not ctx_str[-1].isspace():
                    ctx_str = ctx_str + " "
                    ctx_ids = self.tok_encode(ctx_str)

                # truncate from left
                total = len(ctx_ids) + len(cont_ids)
                if total > self.max_length:
                    overflow = total - self.max_length
                    ctx_ids = ctx_ids[overflow:]

                if len(cont_ids) == 0:
                    batch_out.append((0.0, True))  # edge case
                    continue

                ctx_tensors.append(torch.tensor(ctx_ids, dtype=torch.long))
                cont_tensors.append(torch.tensor(cont_ids, dtype=torch.long))
                metas.append((len(ctx_ids), len(cont_ids)))

            if not metas:
                # all continuations were empty; just extend and move on
                results.extend(batch_out)
                continue

            # ---- 1) first-token logp ----
            ctx_pad = pad_sequence(ctx_tensors, batch_first=True, padding_value=pad_id).to(dev)  # (B, Tc_max)
            logits_ctx, _ = self.model(ctx_pad, targets=None, return_logits=True, full_sequence = True)               # (B, Tc_max, V)
            logp_ctx = F.log_softmax(logits_ctx, dim=-1)

            B = len(metas)
            last_ctx_pos = torch.tensor([lc - 1 for lc, _ in metas], device=dev)
            first_tok_ids = torch.tensor([cont_tensors[i][0].item() for i in range(B)], device=dev)

            first_logp = logp_ctx[torch.arange(B, device=dev), last_ctx_pos, first_tok_ids]
            greedy_first = (logp_ctx[torch.arange(B, device=dev), last_ctx_pos].argmax(-1) == first_tok_ids)

            # ---- 2) remaining tokens ----
            tok_lists, tgt_lists = [], []
            start_idx, end_idx = [], []
            for i, (lc, lcont) in enumerate(metas):
                tokens = torch.cat([ctx_tensors[i], cont_tensors[i]], 0)
                tok_lists.append(tokens[:-1])   # inputs
                tgt_lists.append(tokens[1:])    # targets
                # cont[1:] in target indices -> positions lc .. lc + lcont - 2  (since first cont token is skipped)
                s = lc        # inclusive
                e = lc + lcont - 1  # exclusive
                start_idx.append(s)
                end_idx.append(e)

            inp_pad = pad_sequence(tok_lists, batch_first=True, padding_value=pad_id).to(dev)  # (B, Lmax)
            tgt_pad = pad_sequence(tgt_lists, batch_first=True, padding_value=-1).to(dev)      # (B, Lmax)

            logits_full, _ = self.model(inp_pad, targets=None, return_logits=True, full_sequence = True)             # (B, Lmax, V)
            logp_full = F.log_softmax(logits_full, dim=-1)

            # SAFE gather: replace -1 with 0 (valid index), then mask
            tgt_safe = tgt_pad.clone()
            tgt_safe[tgt_safe < 0] = 0
            gathered = logp_full.gather(2, tgt_safe.unsqueeze(-1)).squeeze(-1)  # (B, Lmax)

            # mask for valid target positions
            valid_mask = tgt_pad != -1  # (B, Lmax)

            # mask for cont[1:] positions
            Lmax = inp_pad.size(1)
            pos = torch.arange(Lmax, device=dev).unsqueeze(0).expand(B, -1)
            mask_rest = torch.zeros_like(valid_mask)
            for i, (s, e) in enumerate(zip(start_idx, end_idx)):
                if e > s:
                    mask_rest[i, s:e] = True

            mask_rest = mask_rest & valid_mask
            logp_rest = (gathered * mask_rest).sum(dim=1)

            pred_ids = logp_full.argmax(dim=-1)
            greedy_rest = torch.where(mask_rest, pred_ids == tgt_pad, torch.ones_like(pred_ids, dtype=torch.bool))
            greedy_all = greedy_first & greedy_rest.all(dim=1)

            totals = (first_logp + logp_rest).tolist()
            greeds = greedy_all.tolist()

            # ---- weave back with empty-cont cases ----
            idx = 0
            for (ctx_str, cont_str), ctx_ids, cont_ids in batch:
                if len(cont_ids) == 0:
                    # already inserted
                    continue
                batch_out.append((totals[idx], greeds[idx]))
                idx += 1

            results.extend(batch_out)

        return results


    @torch.no_grad()
    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        """
        Compute log-likelihood of a full string (perplexity evaluation). 
        Each request.args is a tuple containing a single string.
        """
        results = []
        for req in requests:
            text = req.args[0]
            # Tokenize the entire text, adding BOS token at start
            tokens = self.tok_encode(text)
            # Prepend BOS/EOS token to simulate beginning-of-sequence context
            tokens = [self.prefix_token_id] + tokens
            # We will iterate through the tokens, taking chunks of length <= max_length.
            total_logprob = 0.0
            # We will predict each token (except the first BOS) in sequence.
            # Use a sliding window with overlap=1: each chunk shares the last token of the previous chunk as context.
            start_idx = 0
            while start_idx < len(tokens) - 1:
                end_idx = min(start_idx + self.max_length, len(tokens))
                # Chunk from start_idx to end_idx (inclusive end_idx for input)
                chunk = tokens[start_idx:end_idx]
                # If this is not the first chunk, prepend the previous token (overlap 1) 
                # Actually, by construction start_idx after first iteration equals last index of previous chunk
                # so chunk[0] is already the overlap token from previous iteration.
                # Prepare input and target:
                # Input: all but the last token of chunk
                inp_chunk = chunk[:-1]
                tgt_chunk = chunk[1:]  # targets are the next tokens
                # Convert to tensors
                inp_tensor = torch.tensor(inp_chunk, dtype=torch.long, device=self._device).unsqueeze(0)
                tgt_tensor = torch.tensor(tgt_chunk, dtype=torch.long, device=self._device).unsqueeze(0)
                # Run model
                logits, _ = self.model(inp_tensor, targets=tgt_tensor, return_logits=True, full_sequence = True)
                logits = logits.float()
                log_probs = F.log_softmax(logits, dim=-1)[0]  # shape (chunk_len-1, vocab)
                # Sum log-probs for each target token in this chunk
                # (Ignoring the fact that the first token of chunk is just overlap context)
                # Actually, tgt_chunk length = len(inp_chunk), covering predictions for each inp position.
                for i, target_token in enumerate(tgt_chunk):
                    # i corresponds to prediction for inp_chunk[i]
                    # We want to skip if i < overlap (for first token of chunk after first chunk).
                    if start_idx == 0 or i > 0:
                        # If first chunk, all tokens except BOS are predicted.
                        # If later chunk, skip i=0 because that's the overlapped token prediction (already scored).
                        total_logprob += float(log_probs[i, target_token].item())
                # Move window: overlap by 1 token (last token of this chunk becomes first of next)
                start_idx = end_idx - 1
            results.append((total_logprob,))
        return results

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm=False):
        """
        Generate text for each request until a stop sequence is encountered or max tokens generated.
        Each request.args is a tuple: (context_str, generation_kwargs_dict).
        """
        generations = []
        for req in requests:
            context, gen_kwargs = req.args
            # Extract generation parameters
            max_new_tokens = gen_kwargs.get("max_gen_toks", 256)
            stop_sequences = gen_kwargs.get("until", [])
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            # Always include the EOS token string as a stop (to avoid endlessly generating)
            eos_token_id = self.eot_token_id
            eos_token_str = None
            if eos_token_id is not None:
                try:
                    eos_token_str = self.tokenizer.decode([eos_token_id], clean_up_tokenization_spaces=False)
                except Exception:
                    eos_token_str = None
                if eos_token_str:
                    stop_sequences.append(eos_token_str)
            generated_tokens = []
            generated_text = ""
            # Greedy generation loop
            for _ in range(max_new_tokens):
                # Prepare input IDs (context + already generated tokens)
                input_text = context + self.tokenizer.decode(generated_tokens, clean_up_tokenization_spaces=False)
                input_ids = self.tok_encode(input_text)
                # If input too long, truncate from left (keep last max_length tokens)
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[-self.max_length:]
                inp = torch.tensor(input_ids, dtype=torch.long, device=self._device).unsqueeze(0)
                # Get next token logits from model (since generating, use targets=None to get only last logit)
                logits, _ = self.model(inp, targets=None, return_logits=True)
                logits = logits.float()
                next_token_id = int(logits[0, -1].argmax())
                # If EOS token generated, stop
                if eos_token_id is not None and next_token_id == eos_token_id:
                    break
                # Append token and decode to text
                generated_tokens.append(next_token_id)
                token_str = self.tokenizer.decode([next_token_id], clean_up_tokenization_spaces=False)
                generated_text += token_str
                # Check stop sequences
                combined_text = context + generated_text
                stop_triggered = False
                for stop_seq in stop_sequences:
                    if stop_seq == "":
                        continue
                    idx = combined_text.find(stop_seq)
                    if idx != -1 and idx >= len(context):
                        # Stop sequence found in generated part (or overlaps end of context)
                        stop_triggered = True
                        # Trim generation up to the start of stop sequence
                        gen_cutoff = idx - len(context)
                        generated_text = generated_text[:gen_cutoff]
                        break
                if stop_triggered:
                    break
            generations.append(generated_text)
        return generations
    

        
    