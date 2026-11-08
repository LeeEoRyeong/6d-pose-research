"""PBR criterion is delegated to the baseline unchanged; weak gates are explicit."""
import torch


def _weights(history, axis, device, dtype):
    row=history[axis]
    if row is None:
        return torch.ones(8, device=device, dtype=dtype)
    row=row.detach().to(device=device,dtype=dtype)
    return torch.exp(torch.minimum(row, .51-row)*3).detach()


def weak_terms(pred_front,pred_back,pred_mask,gt_front,gt_back,u_front,u_back,q_pos,q_neg,pbr_criterion):
    """Gate: real mask [N,H,W], HCCE [N,24,H,W], gate binary [N,H,W]."""
    target_shape=pred_mask.shape[:1]+pred_mask.shape[2:]
    for gate in (u_front,u_back,q_pos,q_neg):
        if gate.shape!=target_shape:raise ValueError('gate shape mismatch')
        if not torch.all((gate==0)|(gate==1)):raise ValueError('non-binary gate')
    if torch.any((q_pos>0)&(q_neg>0)):raise ValueError('overlapping positive and negative mask gates')
    if pred_front.shape!=gt_front.shape or pred_back.shape!=gt_back.shape or pred_front.shape[1]!=24:raise ValueError('code shape mismatch')
    zero=pred_mask.sum()*0+pred_front.sum()*0+pred_back.sum()*0
    xyz=zero
    for pred,gt,gate,history in ((pred_front,gt_front,u_front,pbr_criterion.weight_front_error_ratio),(pred_back,gt_back,u_back,pbr_criterion.weight_back_error_ratio)):
        g=gate[:,None,:,:].to(dtype=pred.dtype).detach(); n=g.sum()
        if n.item()==0:continue
        for k in range(3):
            diff=torch.abs(pred[:,k*8:(k+1)*8]-(gt[:,k*8:(k+1)*8]*2-1))*g
            per_bit=diff.sum(dim=(0,2,3))/n
            w=_weights(history,k,pred.device,pred.dtype)
            xyz=xyz+3*(per_bit*w).sum()/w.sum()
    valid=(q_pos+q_neg).to(dtype=pred_mask.dtype).detach()
    n=valid.sum();mask=zero
    if n.item()>0:
        mask=(torch.abs(torch.sigmoid(pred_mask[:,0])-q_pos)*valid).sum()/n
    return {'xyz':xyz,'mask':mask,'u_front':int(u_front.sum().item()),'u_back':int(u_back.sum().item()),'mask_valid':int(valid.sum().item())}


def total_loss(pbr_criterion,pbr_out,pbr_targets,real_out,real_targets,lambda_xyz=.25,lambda_mask=.25):
    """Only PBR tensors enter criterion/history; no real pixels update history."""
    pm,pc=pbr_out; m,f,b=pbr_targets
    parts=pbr_criterion(pc[:,:24],pc[:,24:],pm,f,b,m)
    pbr=3*parts['Front_L1Losses'].sum()+3*parts['Back_L1Losses'].sum()+parts['mask_loss']
    rm,rc=real_out; t=real_targets
    weak=weak_terms(rc[:,:24],rc[:,24:],rm,t['front'],t['back'],t['u_front'],t['u_back'],t['q_pos'],t['q_neg'],pbr_criterion)
    return pbr+lambda_xyz*weak['xyz']+lambda_mask*weak['mask'],pbr,weak
