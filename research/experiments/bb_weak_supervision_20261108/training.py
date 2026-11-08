"""3 PBR + 1 real step primitive. Intentionally no production training entrypoint."""
import torch
from weak_loss import total_loss

def mixed_step(net, optimizer, pbr_criterion, pbr_batch, real_batch, weights=(.25,.25), allow_empty_for_smoke=False):
    x,mask,front,back=pbr_batch
    if x.shape[0]!=3 or real_batch['rgb'].shape[0]!=1:raise ValueError('expected 3 PBR + 1 real')
    usable=sum(int(real_batch[k].sum().item()) for k in ('u_front','u_back','q_pos','q_neg'))
    if not usable and not allow_empty_for_smoke:raise RuntimeError('NO-GO: no trustworthy real supervision')
    if x.device.type!='cpu':raise RuntimeError('GPU training is disabled in this smoke implementation')
    optimizer.zero_grad(set_to_none=True)
    pm,pc=net(torch.cat([x,real_batch['rgb']],dim=0))
    total,pbr,weak=total_loss(pbr_criterion,(pm[:3],pc[:3]),(mask,front,back),(pm[3:],pc[3:]),real_batch,*weights)
    if not torch.isfinite(total):raise RuntimeError('nonfinite loss')
    total.backward();optimizer.step()
    return {'total':float(total.detach()),'pbr':float(pbr.detach()),'real_xyz':float(weak['xyz'].detach()),'real_mask':float(weak['mask'].detach()),'counts':{k:weak[k] for k in ('u_front','u_back','mask_valid')}}
