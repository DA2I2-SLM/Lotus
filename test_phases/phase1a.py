import torch
import torch.nn.functional as F

def _module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.norm(2).item() ** 2)
    return total ** 0.5

def _module_param_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        d = p.detach()
        total += float(d.norm(2).item() ** 2)
    return total ** 0.5

def _print_tensor_stats(name: str, x: torch.Tensor, max_items: int = 0):
    x_det = x.detach()
    print(f"{name}: shape={tuple(x_det.shape)} dtype={x_det.dtype} device={x_det.device}")
    if x_det.numel() > 0 and x_det.is_floating_point():
        print(f"  mean={x_det.mean().item():.6f} std={x_det.std(unbiased=False).item():.6f} "
              f"min={x_det.min().item():.6f} max={x_det.max().item():.6f}")
    if max_items > 0:
        flat = x_det.flatten()
        n = min(max_items, flat.numel())
        print(f"  head({n})={flat[:n].cpu().tolist()}")

def _unpack_pad_apply_ret(ret):
    """
    Supports both:
      (inputs_embeds, attention_mask, token_ids, ts_mask, ts_patch_targets_shifted, ts_patch_valid, ts_chan_id)
    and your older:
      (inputs_embeds, attention_mask, token_ids, ts_mask, ts_segments)
    """
    if not isinstance(ret, (tuple, list)):
        raise ValueError(f"pad_and_apply_batch must return tuple/list, got {type(ret)}")

    if len(ret) == 7:
        inputs_embeds, attention_mask, token_ids, ts_mask, ts_targets, ts_valid, ts_chan_id = ret
        return inputs_embeds, attention_mask, token_ids, ts_mask, ts_targets, ts_valid, ts_chan_id

    if len(ret) == 5:
        inputs_embeds, attention_mask, token_ids, ts_mask, ts_segments = ret

        # Convert ts_segments list-of-dicts into stacked tensors if possible
        if ts_segments is None or len(ts_segments) == 0:
            return inputs_embeds, attention_mask, token_ids, ts_mask, None, None, []

        # Expect each seg has "targets": [n_pred, patch_len], "valid": [n_pred]
        targets_list = []
        valid_list = []
        chan_id_list = []
        n_pred0 = ts_segments[0]["targets"].shape[0]
        patch0 = ts_segments[0]["targets"].shape[1]

        for k, seg in enumerate(ts_segments):
            t = seg["targets"]
            v = seg["valid"]
            if t.shape[0] != n_pred0 or t.shape[1] != patch0:
                raise ValueError(
                    f"Ragged ts_segments not supported by this probe. "
                    f"seg0 targets={n_pred0}x{patch0}, seg{k} targets={t.shape}"
                )
            targets_list.append(t)
            valid_list.append(v)
            chan_id_list.append(int(seg.get("chan_id", 0)))

        ts_targets = torch.stack([t.to(inputs_embeds.device) for t in targets_list], dim=0)
        ts_valid = torch.stack([v.to(inputs_embeds.device) for v in valid_list], dim=0)
        return inputs_embeds, attention_mask, token_ids, ts_mask, ts_targets, ts_valid, chan_id_list

    raise ValueError(f"Unexpected pad_and_apply_batch return length: {len(ret)}")

@torch.no_grad()
def preview_ts_generation(
    model,
    batch,
    n_show_patches: int = 5,
    n_show_segments: int = 1,
):
    """
    Forward-only: prints what the TS path generates (pred vs gt for a few patches).
    """
    model.eval()

    ret = model.pad_and_apply_batch(batch, return_targets=True)
    inputs_embeds, attention_mask, token_ids, ts_mask, ts_targets, ts_valid, ts_chan_id = _unpack_pad_apply_ret(ret)

    print("\n=== preview_ts_generation ===")
    _print_tensor_stats("inputs_embeds", inputs_embeds)
    _print_tensor_stats("attention_mask", attention_mask)

    if ts_targets is None or ts_targets.numel() == 0:
        print("No TS targets found in this batch.")
        return

    print(f"ts_targets: shape={tuple(ts_targets.shape)} dtype={ts_targets.dtype}")
    print(f"ts_valid:   shape={tuple(ts_valid.shape)} dtype={ts_valid.dtype}")
    print(f"ts_chan_id: len={len(ts_chan_id)} head={ts_chan_id[:min(5,len(ts_chan_id))]}")

    # LLM forward
    out = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = out.hidden_states[-1]  # [B, L, H]

    # TS hidden vectors at positions marked by ts_mask
    ts_hidden = last_hidden[ts_mask]  # [N_ts_total*N_pred, H] if everything is aligned

    N_ts_total, N_pred, patch_size = ts_targets.shape
    expected = N_ts_total * N_pred
    print(f"ts_hidden rows={ts_hidden.shape[0]} expected={expected}")
    if ts_hidden.shape[0] != expected:
        print("WARNING: ts_mask count does not match N_ts_total*N_pred. Your ts_mask/targets alignment is off.")
        return

    ts_hidden = model.decoder_projector(ts_hidden.float())
    ts_hidden_3d = ts_hidden.reshape(N_ts_total, N_pred, -1)

    pred = model.decoder(ts_hidden_3d, ts_chan_id)  # [N_ts_total, N_pred, patch_size] (should)
    print(f"pred: shape={tuple(pred.shape)} dtype={pred.dtype}")

    if pred.shape[-1] != patch_size:
        print(f"WARNING: pred patch_size={pred.shape[-1]} but target patch_size={patch_size}. Patch sizing mismatch.")
        return

    # Apply valid mask and compute quick metrics
    valid_flat = ts_valid.reshape(-1)
    gt_flat = ts_targets.reshape(-1, patch_size)
    pred_flat = pred.reshape(-1, patch_size)

    gt_v = gt_flat[valid_flat]
    pred_v = pred_flat[valid_flat]

    mse = F.mse_loss(pred_v, gt_v).item() if gt_v.numel() > 0 else float("nan")
    mae = (pred_v - gt_v).abs().mean().item() if gt_v.numel() > 0 else float("nan")
    print(f"valid patches: {gt_v.shape[0]} | MSE={mse:.6f} | MAE={mae:.6f}")

    # Show a few patches for first few segments
    segs_to_show = min(n_show_segments, N_ts_total)
    for s in range(segs_to_show):
        print(f"\n--- Segment {s} (chan_id={ts_chan_id[s] if len(ts_chan_id)>s else 'NA'}) ---")
        for p in range(min(n_show_patches, N_pred)):
            if not bool(ts_valid[s, p].item()):
                continue
            gt = ts_targets[s, p].detach().cpu().tolist()
            pr = pred[s, p].detach().cpu().tolist()
            print(f"patch {p}:")
            print(f"  gt:  {gt}")
            print(f"  pred:{pr}")

def probe_ts_training_step(
    model,
    batch,
    alpha: float = None,
    print_topk_params: int = 0,
):
    """
    Runs forward + backward once and reports gradient norms for:
      - encoder_projector
      - decoder_projector
      - decoder (and decoder_head if present)
    Also prints simple TS prediction stats.

    Call this inside training (or standalone) to verify TS modules actually get gradients.
    """
    model.train()
    if alpha is None:
        alpha = getattr(model, "recon_alpha", 1.0)

    # zero grads
    model.zero_grad(set_to_none=True)

    ret = model.pad_and_apply_batch(batch, return_targets=True)
    inputs_embeds, attention_mask, token_ids, ts_mask, ts_targets, ts_valid, ts_chan_id = _unpack_pad_apply_ret(ret)

    out = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    # NTP loss (robust masking)
    logits = out.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = token_ids[:, 1:].contiguous().clone()
    shift_labels[ts_mask[:, 1:]] = -100

    ntp_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    recon_loss = torch.tensor(0.0, device=inputs_embeds.device)
    if ts_targets is not None and ts_targets.numel() > 0:
        last_hidden = out.hidden_states[-1]
        ts_hidden = last_hidden[ts_mask]

        N_ts_total, N_pred, patch_size = ts_targets.shape
        expected = N_ts_total * N_pred
        if ts_hidden.shape[0] != expected:
            raise RuntimeError(
                f"ts_hidden rows={ts_hidden.shape[0]} expected={expected}. "
                "ts_mask/targets alignment is wrong."
            )

        ts_hidden = model.decoder_projector(ts_hidden.float())
        ts_hidden_3d = ts_hidden.reshape(N_ts_total, N_pred, -1)

        pred = model.decoder(ts_hidden_3d, ts_chan_id)
        if pred.shape[-1] != patch_size:
            raise RuntimeError(
                f"pred patch_size={pred.shape[-1]} target patch_size={patch_size}. Patch sizing mismatch."
            )

        valid_flat = ts_valid.reshape(-1)
        gt_flat = ts_targets.reshape(-1, patch_size)
        pred_flat = pred.reshape(-1, patch_size)

        if valid_flat.any():
            recon_loss = F.mse_loss(pred_flat[valid_flat], gt_flat[valid_flat].to(pred_flat.dtype))
        else:
            recon_loss = torch.tensor(0.0, device=inputs_embeds.device)

    total_loss = ntp_loss + alpha * recon_loss
    total_loss.backward()

    print("\n=== probe_ts_training_step ===")
    print(f"total_loss={total_loss.item():.6f} ntp_loss={ntp_loss.item():.6f} recon_loss={recon_loss.item():.6f} alpha={alpha}")

    # gradient norms
    if hasattr(model, "encoder_projector"):
        print(f"encoder_projector: param_norm={_module_param_norm(model.encoder_projector):.6f} grad_norm={_module_grad_norm(model.encoder_projector):.6f}")
    if hasattr(model, "decoder_projector"):
        print(f"decoder_projector: param_norm={_module_param_norm(model.decoder_projector):.6f} grad_norm={_module_grad_norm(model.decoder_projector):.6f}")
    if hasattr(model, "decoder"):
        print(f"decoder:           param_norm={_module_param_norm(model.decoder):.6f} grad_norm={_module_grad_norm(model.decoder):.6f}")
    if hasattr(model, "decoder_head"):
        print(f"decoder_head:      param_norm={_module_param_norm(model.decoder_head):.6f} grad_norm={_module_grad_norm(model.decoder_head):.6f}")

    # show a tiny preview (first segment, first 3 valid patches)
    if ts_targets is not None and ts_targets.numel() > 0 and recon_loss.item() > 0:
        with torch.no_grad():
            valid_flat = ts_valid.reshape(-1)
            N_ts_total, N_pred, patch_size = ts_targets.shape
            out2 = model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = out2.hidden_states[-1]
            ts_hidden = last_hidden[ts_mask]
            ts_hidden = model.decoder_projector(ts_hidden.float()).reshape(N_ts_total, N_pred, -1)
            pred = model.decoder(ts_hidden, ts_chan_id)

            shown = 0
            print("\nPreview first segment patches (gt vs pred):")
            for p in range(N_pred):
                if not bool(ts_valid[0, p].item()):
                    continue
                gt = ts_targets[0, p].detach().cpu().tolist()
                pr = pred[0, p].detach().cpu().tolist()
                print(f"patch {p}: gt={gt} pred={pr}")
                shown += 1
                if shown >= 3:
                    break

    if print_topk_params > 0:
        # Optional: show which params got the biggest grad norms
        grads = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            grads.append((float(p.grad.detach().norm().item()), name, tuple(p.shape)))
        grads.sort(reverse=True, key=lambda x: x[0])
        print("\nTop grad norms:")
        for g, n, sh in grads[:print_topk_params]:
            print(f"  {g:.6f} | {n} | {sh}")

    return total_loss.item(), ntp_loss.item(), recon_loss.item()