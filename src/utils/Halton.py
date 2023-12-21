import torch

def HaltonInt(Index : int, Base : int) ->float: 
	Result = 0.0
	InvBase = 1.0 / Base
	Fraction = InvBase
	while (Index > 0):
		Result += (Index % Base) * Fraction
		Index /= Base
		Fraction *= InvBase
	return max(0.0, min(1.0, Result))

def HaltonTensor(Index : torch.Tensor, Base : int) ->torch.Tensor: 
    Index = Index.float()
    Result = torch.zeros_like(Index, dtype=float)
    unfinished = torch.ones_like(Index, dtype=torch.bool)
    InvBase = 1.0 / Base
    Fraction = InvBase
    while (torch.sum(unfinished) > 0):
        Result += unfinished.float() * (Index % Base) * Fraction
        Index /= Base
        Fraction *= InvBase
        unfinished = (Index > 0)
    return Result