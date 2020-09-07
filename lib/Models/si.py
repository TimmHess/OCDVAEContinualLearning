import torch
import torch.nn as nn


class SI_StorageUnit(nn.Module):
    """
    An empty module, used to store SI buffers
    """
    def __init__(self):
        super(SI_StorageUnit, self).__init__()

        self.epsilon = 1e-3
        # Init SI containers
        self.W = {}
        self.p_old = {}

        # Flag set to true after first task training is complete
        self.is_initialized = False

        return


def update_omega(model, storage_unit, W, epsilon):
    '''After completing training on a task, update the per-parameter regularization strength.
    [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
    [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

    # Loop over all parameters
    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')

            # Find/calculate new values for quadratic penalty on parameters
            #p_prev = getattr(model, '{}_SI_prev_task'.format(n))
            p_prev = getattr(storage_unit, '{}_SI_prev_task'.format(n))
            p_current = p.detach().clone() # detach ? maybe breaks loss?
            p_change = p_current - p_prev # original
            #p_change = p_prev - p_current # test

            omega_add = W[n]/(p_change**2 + epsilon)
            try:
                #omega = getattr(model, '{}_SI_omega'.format(n))
                omega = getattr(storage_unit, '{}_SI_omega'.format(n))
                # Copy omega to parameter shape
                new_omega = torch.zeros_like(p)
                new_omega[:omega.shape[0]] = omega
                omega = new_omega
                omega_new = omega + omega_add
            except AttributeError:
                #omega = p.detach().clone().zero_()
                #omega_new = omega + (0.25 * omega_add) 
                omega_new = omega_add
            #print("")
            #print("updated omega..")
            #print("min", torch.min(omega_new), "max", torch.max(omega_new))
            # Store these new values in the model
            #model.register_buffer('{}_SI_prev_task'.format(n), p_current)
            storage_unit.register_buffer('{}_SI_prev_task'.format(n), p_current)
            #model.register_buffer('{}_SI_omega'.format(n), omega_new)
            storage_unit.register_buffer('{}_SI_omega'.format(n), omega_new)


def surrogate_loss(model, storage_unit):
    '''Calculate SI's surrogate loss. '''
    try:
        losses = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                n = n.replace('.', '__')
                #prev_values = getattr(model, '{}_SI_prev_task'.format(n))
                prev_values = getattr(storage_unit, '{}_SI_prev_task'.format(n))
                #omega = getattr(model, '{}_SI_omega'.format(n))
                omega = getattr(storage_unit, '{}_SI_omega'.format(n))
                # Calculate SI's surrogate loss, sum over all parameters
                #if("outc" in n): # ignore recently added output neurons
                #    losses.append((omega * (p[:prev_values.shape[0]]-prev_values)**2).sum())
                #else:
                if p.size(0) > omega.size(0):
                    tmp_omega = torch.zeros_like(p)
                    tmp_omega[:omega.shape[0]] = omega
                    losses.append((tmp_omega * ((p-prev_values)**2)).sum())
                    #losses.append((((p-prev_values)**2)).sum())
                else:
                #losses.append((omega * ((p-prev_values)**2)[:omega.shape[0]]).sum())
                    losses.append((omega * ((p-prev_values)**2)).sum())
                    #losses.append((((p-prev_values)**2)).sum())
        #print("sum_losses", sum(losses))
        return sum(losses)
    except AttributeError:
        # SI-loss is 0 if there is no stored omega yet
        return torch.tensor(0., device=storage_unit._device())


# Register starting param-values (needed for "intelligent synapses").
def register_si_params(model, storage_unit):
    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            storage_unit.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())
            #model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())

def update_registered_si_params(model, storage_unit):
    """
    Updates stored buffers that need to grow with the model
    """
    for n, p in model.named_parameters():
        if(p.requires_grad):
            n = n.replace('.', '__')
            #if("outc" in n):
                # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
                # n = scores.size(1)
                # if n > target_scores.size(1):
                #     n_batch = scores.size(0)
                #     zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
                #     zeros_to_add = zeros_to_add.to(device)
                #     targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

            # Get buffer according to p
            buffer = getattr(storage_unit, '{}_SI_prev_task'.format(n))
            # Check if update is needed because shape of p changed
            if buffer.size(0) < p.size(0):
                print("buffer shapre", buffer.shape)
                print("p shape", p.shape)
            
                # Get new zero buffer
                new_buffer = torch.zeros_like(p)
                # Copy buffer to new buffer
                new_buffer[:buffer.shape[0]] = buffer
                #print("new buffer shape", new_buffer.shape)
                storage_unit.register_buffer('{}_SI_prev_task'.format(n), new_buffer.data.clone())
    return

# Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
def init_si_params(model, storage_unit):
    storage_unit.W = {}
    storage_unit.p_old = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            storage_unit.W[n] = p.data.clone().zero_()
            storage_unit.p_old[n] = p.data.clone()

# Update running parameter importance estimates in W
def update_si_parameters(model, storage_unit):
    #if isinstance(model, ContinualLearner) and (model.si_c>0):
    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            if p.grad is not None:
                storage_unit.W[n].add_(-p.grad*(p.detach()-storage_unit.p_old[n])) #original
                #storage_unit.W[n].add_(p.grad*(storage_unit.p_old[n]-p.detach())) # test
            storage_unit.p_old[n] = p.detach().clone()

# SI: calculate and update the normalized path integral
def update_si_integral(model, storage_unit):
    #if isinstance(model, ContinualLearner) and (model.si_c>0):
    update_omega(model, storage_unit, storage_unit.W, storage_unit.epsilon)
    storage_unit.is_initialized = True